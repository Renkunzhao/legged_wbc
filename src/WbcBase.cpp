//
// Created by Kunzhao on 2025/8/31.
//
#include <cstddef>
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "legged_wbc/Utils.h"
#include "legged_wbc/Task.h"
#include "legged_wbc/Types.h"
#include "legged_wbc/Lie.h"
#include "legged_wbc/WbcBase.h"

#include <logger/CsvLogger.h>

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/math/rpy.hpp>

namespace fs = std::filesystem;
using namespace Lie;

namespace legged {

vector_t WbcBase::update(LeggedState des_state, LeggedState real_state, std::array<bool, 4> contactFlag,
                         scalar_t /*period*/ , std::string /*method*/) {
  if(verbose_) {
    std::cout << "[WbcBase] contactFlag:\n" << contactFlag[0] << " " << contactFlag[1] << " " << contactFlag[2] << " " << contactFlag[3] << std::endl;
  }

  contactFlag_ = contactFlag;
  numContacts_ = std::accumulate(contactFlag_.begin(), contactFlag_.end(), 0);

  des_state_ = des_state;
  real_state_ = real_state;

  qDesired_ = des_state_.custom_state("q_pin");
  vDesired_ = des_state_.custom_state("v_pin");
  fDesired_ = des_state_.custom_state("f_pin");
  comDes_ = des_state_.com_pos();
  vcomDes_ = des_state_.com_vel_W();
  hgDes_ << des_state_.com_lin_mom_W(), des_state_.com_ang_mom_W();
  qMeasured_ = real_state_.custom_state("q_pin");
  vMeasured_ = real_state_.custom_state("v_pin");
  updateMeasured();
  updateDesired();
  return {};
}

void WbcBase::updateMeasured() {
  const auto& model = leggedModel_.model();
  auto& data = leggedModel_.data();

  // EOM Task
  MMeasured_ = matrix_t(leggedModel_.nDof(), leggedModel_.nDof());
  nleMeasured_ = vector_t(leggedModel_.nDof());
  MMeasured_ = pinocchio::crba(model, data, qMeasured_);
  MMeasured_.triangularView<Eigen::StrictlyLower>() = MMeasured_.transpose().triangularView<Eigen::StrictlyLower>();
  nleMeasured_ = pinocchio::nonLinearEffects(model, data, qMeasured_, vMeasured_);

  pinocchio::forwardKinematics(model, data, qMeasured_, vMeasured_);

  // EOM Task & SwingLegTask & NoContactMotionTask
  pinocchio::computeJointJacobians(model, data);
  jMeasured_ = matrix_t(3 * leggedModel_.nContacts3Dof(), leggedModel_.nDof());
  jMeasured_ = leggedModel_.jacobian3Dof(qMeasured_);

  // SwingLegTask & NoContactMotionTask
  pinocchio::computeJointJacobiansTimeVariation(model, data, qMeasured_, vMeasured_);
  djMeasured_ = matrix_t(3 * leggedModel_.nContacts3Dof(), leggedModel_.nDof());
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, leggedModel_.nDof());
    pinocchio::getFrameJacobianTimeVariation(model, data, leggedModel_.contact3DofIds()[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
    djMeasured_.block(3 * i, 0, 3, leggedModel_.nDof()) = jac.template topRows<3>();
  }

  // ComTask
  pinocchio::computeCentroidalMomentum(model, data, qMeasured_, vMeasured_);
  comAct_ = data.com[0];
  vcomAct_ = data.vcom[0];
  hgAct_ = data.hg.toVector();

  AMeasured_ = matrix_t(6, leggedModel_.nDof());
  dAMeasured_ = matrix_t(6, leggedModel_.nDof());
  pinocchio::dccrba(model, data, qMeasured_, vMeasured_);
  AMeasured_ = data.Ag;
  dAMeasured_ = data.dAg;

  if(verbose_) {
    std::cout << "[WbcBase] MMeasured:\n" << MMeasured_ << std::endl;
    std::cout << "[WbcBase] nleMeasured:" << nleMeasured_.transpose() << std::endl;
    std::cout << "[WbcBase] jMeasured:\n" << jMeasured_ << std::endl;
    std::cout << "[WbcBase] djMeasured:\n" << djMeasured_ << std::endl;
    // std::cout << "[WbcBase] data.Ag: rows = " << data.Ag.rows() << " cols = " << data.Ag.cols() << std::endl;
    // std::cout << "[WbcBase] data.dAg: rows = " << data.dAg.rows() << " cols = " << data.dAg.cols() << std::endl;
  }
}

void WbcBase::updateDesired() {
  const auto& model = leggedModel_.model();
  auto& data = leggedModel_.data();
}

Task WbcBase::formulateFloatingBaseEomTask() {
  matrix_t s(leggedModel_.nJoints(), leggedModel_.nDof());
  s.block(0, 0, leggedModel_.nJoints(), 6).setZero();
  s.block(0, 6, leggedModel_.nJoints(), leggedModel_.nJoints()).setIdentity();

  matrix_t a = (matrix_t(leggedModel_.nDof(), numDecisionVars_) << MMeasured_, -jMeasured_.transpose(), -s.transpose()).finished();
  vector_t b = -nleMeasured_;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] FloatingBaseEomTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }

  return {a, b, matrix_t(), vector_t()};
}

Task WbcBase::formulateTorqueLimitsTask() {
  matrix_t d(2 * leggedModel_.nJoints(), numDecisionVars_);
  d.setZero();
  matrix_t i = matrix_t::Identity(leggedModel_.nJoints(), leggedModel_.nJoints());
  d.block(0, leggedModel_.nDof() + 3 * leggedModel_.nContacts3Dof(), leggedModel_.nJoints(), leggedModel_.nJoints()) = i;
  d.block(leggedModel_.nJoints(), leggedModel_.nDof() + 3 * leggedModel_.nContacts3Dof(), leggedModel_.nJoints(),
          leggedModel_.nJoints()) = -i;
  vector_t f(2 * leggedModel_.nJoints());
  for (size_t l = 0; l < 2 * leggedModel_.nJoints() / 3; ++l) {
    f.segment<3>(3 * l) = torqueLimits_;
  }

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] TorqueLimitsTask " << std::endl;
    std::cout << "[WbcBase] d:\n" << d << std::endl;
    std::cout << "[WbcBase] f: " << f.transpose() << std::endl;
  }

  return {matrix_t(), vector_t(), d, f};
}

Task WbcBase::formulateNoContactMotionTask() {
  matrix_t a(3 * numContacts_, numDecisionVars_);
  vector_t b(a.rows());
  a.setZero();
  b.setZero();
  size_t j = 0;
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); i++) {
    if (contactFlag_[i]) {
      a.block(3 * j, 0, 3, leggedModel_.nDof()) = jMeasured_.block(3 * i, 0, 3, leggedModel_.nDof());
      b.segment(3 * j, 3) = -djMeasured_.block(3 * i, 0, 3, leggedModel_.nDof()) * vMeasured_;
      j++;
    }
  }

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] NoContactMotionTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  noContactMotionTask_ = Task(a, b, matrix_t(), vector_t());
  return noContactMotionTask_;
}

Task WbcBase::formulateNoSlipXYTask() {
  matrix_t a(2 * numContacts_, numDecisionVars_);
  vector_t b(a.rows());
  a.setZero();
  b.setZero();
  size_t j = 0;
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); i++) {
    if (contactFlag_[i]) {
      a.block(2 * j, 0, 2, leggedModel_.nDof()) = jMeasured_.block(3 * i, 0, 2, leggedModel_.nDof());
      b.segment(2 * j, 2) = -djMeasured_.block(3 * i, 0, 2, leggedModel_.nDof()) * vMeasured_;
      j++;
    }
  }

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] NoSlipXYTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  return {a, b, matrix_t(), vector_t()};
}

Task WbcBase::formulateFrictionConeTask() {
  matrix_t a(3 * (leggedModel_.nContacts3Dof() - numContacts_), numDecisionVars_);
  a.setZero();
  size_t j = 0;
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    if (!contactFlag_[i]) {
      a.block(3 * j++, leggedModel_.nDof() + 3 * i, 3, 3) = matrix_t::Identity(3, 3);
    }
  }
  vector_t b(a.rows());
  b.setZero();

  matrix_t frictionPyramic(5, 3);  // clang-format off
  frictionPyramic << 0, 0, -1,
                     1, 0, -frictionCoeff_,
                    -1, 0, -frictionCoeff_,
                     0, 1, -frictionCoeff_,
                     0,-1, -frictionCoeff_;  // clang-format on

  matrix_t d(5 * numContacts_, numDecisionVars_);
  d.setZero();
  j = 0;
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    if (contactFlag_[i]) {
      d.block(5 * j++, leggedModel_.nDof() + 3 * i, 5, 3) = frictionPyramic;
    }
  }
  vector_t f = Eigen::VectorXd::Zero(d.rows());

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] FrictionConeTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
    std::cout << "[WbcBase] d:\n" << d << std::endl;
    std::cout << "[WbcBase] f: " << f.transpose() << std::endl;
  }
  
  return {a, b, d, f};
}

Task WbcBase::formulateBaseAccelTaskPD() {
  matrix_t a(6, numDecisionVars_);
  a.setZero();
  a.block(0, 0, 6, 6) = matrix_t::Identity(6, 6);

  Vector6 pos_error, vel_error, accel, b; 

  // https://github.com/stack-of-tasks/pinocchio/issues/16 pinocchio store quat in [x,y,w,z]
  Eigen::Vector4d quat_des = quat_wxyz(qDesired_.segment(3,4));
  Eigen::Vector4d quat = quat_wxyz(qMeasured_.segment(3,4));
  Eigen::Matrix3d R_des = quat_ToR(quat_des);
  Eigen::Matrix3d R = quat_ToR(quat);
  pos_error << R.transpose() * (qDesired_.head(3) - qMeasured_.head(3)),
                quat_boxminusL(quat_des, quat);

  // Representation-Free Model Predictive Control for Dynamic Motions in Quadrupeds (https://arxiv.org/pdf/2012.10002 p5 equ-29,30) 
  Eigen::Vector3d w_des = vDesired_.segment(3,3);
  Eigen::Vector3d w = vMeasured_.segment(3,3);

  vel_error << R.transpose()*R_des*vDesired_.head(3) - vMeasured_.head(3),
               R.transpose()*R_des*w_des - w; 
  b = wbcParam_.baseAccelKp_.asDiagonal() * pos_error + wbcParam_.baseAccelKd_.asDiagonal() * vel_error;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] BaseAccelTaskPD " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
      
  baseAccTask_ = Task(a, b, matrix_t(), vector_t());
  return baseAccTask_;
}

Task WbcBase::formulateComTask() {
  matrix_t a(6, numDecisionVars_);
  a.setZero();
  a.block(0, 0, 6, leggedModel_.nDof()) = AMeasured_;

  Eigen::Vector4d quat = quat_wxyz(qMeasured_.segment(3,4));
  Eigen::Matrix3d R = quat_ToR(quat);
  Eigen::Vector3d a_com;
  a_com = wbcParam_.comKp_.head(3).asDiagonal()*(comDes_ - comAct_) 
        + wbcParam_.comKd_.head(3).asDiagonal()*(vcomDes_ - vcomAct_);

  Vector6 h_des;
  h_des << mass_*a_com, 
           wbcParam_.comKd_.tail(3).asDiagonal()*(hgDes_ - hgAct_).tail(3);

  Vector6 b;
  b = h_des - dAMeasured_*vMeasured_;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] ComTaskPD " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }

  comTask_ = Task(a, b, matrix_t(), vector_t());
  return comTask_;
}

Task WbcBase::formulateSwingLegTask() {
  std::vector<Eigen::Vector3d> posMeasured = leggedModel_.contact3DofPoss(qMeasured_);
  std::vector<Eigen::Vector3d> velMeasured = leggedModel_.contact3DofVels(qMeasured_, vMeasured_);
  std::vector<Eigen::Vector3d> posDesired = leggedModel_.contact3DofPoss(qDesired_);
  std::vector<Eigen::Vector3d> velDesired = leggedModel_.contact3DofVels(qDesired_, vDesired_);

  matrix_t a(3 * (leggedModel_.nContacts3Dof() - numContacts_), numDecisionVars_);
  vector_t b(a.rows());
  a.setZero();
  b.setZero();
  size_t j = 0;
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    if (!contactFlag_[i]) {
      Eigen::Vector3d accel = wbcParam_.swingKp_ * (posDesired[i] - posMeasured[i]) + wbcParam_.swingKd_ * (velDesired[i] - velMeasured[i]);
      a.block(3 * j, 0, 3, leggedModel_.nDof()) = jMeasured_.block(3 * i, 0, 3, leggedModel_.nDof());
      b.segment(3 * j, 3) = accel - djMeasured_.block(3 * i, 0, 3, leggedModel_.nDof()) * vMeasured_;
      j++;
    }
  }

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] SwingLegTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  swingLegTask_ = Task(a, b, matrix_t(), vector_t());
  return swingLegTask_;
}

Task WbcBase::formulateContactForceTask() {
  // [0 I 0]x = lambda
  matrix_t a(3 * leggedModel_.nContacts3Dof(), numDecisionVars_);
  vector_t b(a.rows());
  a.setZero();

  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    a.block(3 * i, leggedModel_.nDof() + 3 * i, 3, 3) = matrix_t::Identity(3, 3);
  }
  b = fDesired_;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] ContactForceTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  contactForceTask_ = Task(a, b, matrix_t(), vector_t());
  return contactForceTask_;
}

Task WbcBase::formulateSumFzTask(){
  // [0 Sz 0]x = Fz
  matrix_t a = matrix_t::Zero(1, numDecisionVars_);
  vector_t b(a.rows());
  matrix_t Sz(1, 3*leggedModel_.nContacts3Dof());
  Sz << 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1;

  a.block(0, leggedModel_.nDof(), 1, 3*leggedModel_.nContacts3Dof()) = Sz;
  b = Sz*fDesired_;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] SumFzTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  SumFzTask_ = Task(a, b, matrix_t(), vector_t());
  return SumFzTask_;
}


Task WbcBase::formulateJointTorqueTask() {
  matrix_t a = matrix_t::Zero(leggedModel_.nJoints(), numDecisionVars_);
  vector_t b = vector_t::Zero(a.rows());

  a.rightCols(leggedModel_.nJoints()) = matrix_t::Identity(leggedModel_.nJoints(), leggedModel_.nJoints());

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] JointTorqueTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  jointTorqueTask_ = Task(a, b, matrix_t(), vector_t());
  return jointTorqueTask_;
}

void WbcBase::loadWbcParam(const std::string& motionFile, bool verbose)
{
    YAML::Node cfg = YAML::LoadFile(motionFile);
    WbcParameters param;

    if (verbose) {
        std::cout << "[WbcBase] Loading motion parameters from " << motionFile << std::endl;
    }

    param.motionName_ = cfg["motionName"].as<std::string>();

    param.constraintList_ = cfg["constraintList"].as<vector<string>>();

    // === Base Acceleration Task ===
    param.baseAccelKp_ = yamlToEigenVector(cfg["baseAccelTask"]["baseAcc_kp"]);
    param.baseAccelKd_ = yamlToEigenVector(cfg["baseAccelTask"]["baseAcc_kd"]);

    // === COM Acceleration Task ===
    param.comKp_ = yamlToEigenVector(cfg["comTask"]["com_kp"]);
    param.comKd_ = yamlToEigenVector(cfg["comTask"]["com_kd"]);

    // === Swing Leg Task ===
    param.swingKp_ = cfg["swingLegTask"]["kp"].as<double>();
    param.swingKd_ = cfg["swingLegTask"]["kd"].as<double>();

    // === Joint PD ===
    param.jointKp_ = cfg["jointKp"].as<double>();
    param.jointKd_ = cfg["jointKd"].as<double>();

    // === Weight (optional) ===
    if (cfg["weight"]) {
        const auto& w = cfg["weight"];
        param.weightBaseAccel_    = yamlToEigenVector(w["baseAccel"]);
        param.weightCom_ = yamlToEigenVector(w["com"]);
        param.weightContactForce_ = vector_t::Zero(3*leggedModel_.nContacts3Dof());
        param.weightNoContactMotion_ = vector_t::Zero(3*leggedModel_.nContacts3Dof());
        for (size_t i=0; i<leggedModel_.nContacts3Dof(); ++i) {
          param.weightContactForce_.segment(3*i,3) = yamlToEigenVector(w["contactForce"]);
          param.weightNoContactMotion_.segment(3*i,3) = yamlToEigenVector(w["noContactMotion"]);
        }
        param.weightSumFz_        = w["SumFz"].as<double>();
        param.weightSwingLeg_     = w["swingLeg"].as<double>();
        param.weightJointTorque_  = w["jointTorque"].as<double>();
    }

    // === Append to list ===
    wbcParamList_.push_back(param);

    // === Print summary if verbose ===
    if (verbose) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[WbcBase] baseAccelKp: " << param.baseAccelKp_.transpose() << std::endl;
        std::cout << "[WbcBase] baseAccelKd: " << param.baseAccelKd_.transpose() << std::endl;
        std::cout << "[WbcBase] comKp:  " << param.comKp_.transpose() << std::endl;
        std::cout << "[WbcBase] comKd:  " << param.comKd_.transpose() << std::endl;
        std::cout << "[WbcBase] swingKp: " << param.swingKp_
                  << "  swingKd: " << param.swingKd_ << std::endl;
        std::cout << "[WbcBase] jointKp: " << param.jointKp_
                  << "  jointKd: " << param.jointKd_ << std::endl;
        if (cfg["weight"]) {
            std::cout << "[WbcBase] weight.BaseAccel: " << param.weightBaseAccel_.transpose() << std::endl;
            std::cout << "[WbcBase] weight.Com:  " << param.weightCom_.transpose() << std::endl;
            std::cout << "[WeightedWbc] weightContactForce: " << param.weightContactForce_.transpose() << std::endl;
            std::cout << "[WeightedWbc] weightNoContactMotion: " << param.weightNoContactMotion_.transpose() << std::endl;
            std::cout << "[WeightedWbc] weightSumFz: " << param.weightSumFz_ << std::endl;
            std::cout << "[WeightedWbc] weightSwingLeg: " << param.weightSwingLeg_ << std::endl;
            std::cout << "[WeightedWbc] weightJointTorque: " << param.weightJointTorque_ << std::endl;
        }
        std::cout << "[WbcBase] Motion param loaded successfully." << std::endl;
    }
}

void WbcBase::setWbcParam(const std::string& motionName) {
    for (const auto& param : wbcParamList_) {
        if (param.motionName_ == motionName) {
            wbcParam_ = param;
            if (verbose_) {
                std::cout << "[WbcBase] WBC parameters set to motion: " << motionName << std::endl;
            }
            return;
        }
    }
    throw std::runtime_error("[WbcBase] Motion name not found: " + motionName);
}


void WbcBase::loadTasksSetting(const std::string& configFile) {
    std::cout << "[WbcBase] Load config from " << configFile << std::endl;
    YAML::Node configNode = YAML::LoadFile(configFile);

    verbose_ = configNode["verbose"].as<bool>();

    // === General robot setup ===
    leggedModel_.loadUrdf(configNode["urdfPath"].as<std::string>(), "quaternion",
                          configNode["baseName"].as<std::string>(),
                          configNode["contact3DofNames"].as<std::vector<std::string>>(),
                          configNode["contact6DofNames"].as<std::vector<std::string>>(),
                          configNode["hipNames"].as<std::vector<std::string>>(),
                          verbose_);

    mass_ = pinocchio::computeTotalMass(leggedModel_.model());

    numDecisionVars_ = leggedModel_.nDof()
        + 3 * leggedModel_.nContacts3Dof()
        + 6 * leggedModel_.nContacts6Dof()
        + leggedModel_.nJoints();

    qMeasured_.resize(leggedModel_.nqBase());
    vMeasured_.resize(leggedModel_.nDof());
    qDesired_.resize(leggedModel_.nqBase());
    vDesired_.resize(leggedModel_.nDof());
    vDesiredLast_.resize(leggedModel_.nDof());
    fDesired_.resize(3 * leggedModel_.nContacts3Dof() + 6 * leggedModel_.nContacts6Dof());

    // === Load each motion config ===
    if (configNode["motionList"] && configNode["motionList"].IsSequence()) {
        wbcParamList_.clear();

        // Get directory of the main config file
        fs::path baseDir = fs::absolute(fs::path(configFile)).parent_path();

        for (const auto& motionName : configNode["motionList"].as<std::vector<std::string>>()) {
            fs::path motionFile = baseDir / (motionName + ".yaml");

            if (!fs::exists(motionFile)) {
                throw std::runtime_error("[WbcBase] Motion file not found: " + motionFile.string());
            }

            loadWbcParam(motionFile.string(), verbose_);
        }
    } else {
        throw std::runtime_error("[WbcBase] motionList not found or not a list in config file.");
    }

    // === Select the first motion as default ===
    if (!wbcParamList_.empty()) {
        setWbcParam("stand");
    }

    torqueLimits_ = yamlToEigenVector(configNode["torqueLimitsTask"]);
    frictionCoeff_ = configNode["frictionConeTask"]["frictionCoefficient"].as<double>();

    if (verbose_) {
        std::cout << "[WbcBase] Mass: " << mass_ << std::endl;
        std::cout << "[WbcBase] Decision vars: " << numDecisionVars_ << std::endl;
        std::cout << "[WbcBase] torqueLimits: " << torqueLimits_.transpose() << std::endl;
        std::cout << "[WbcBase] frictionCoeff: " << frictionCoeff_ << std::endl;
        std::cout << "[WbcBase] Loaded " << wbcParamList_.size()
                  << " motion params (default = stand)" << std::endl;
    }
}

void WbcBase::log(const vector_t& x){
    CsvLogger& logger = CsvLogger::getInstance();
}

}  // namespace legged
