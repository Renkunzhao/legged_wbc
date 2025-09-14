//
// Created by qiayuan on 2022/7/1.
//
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "legged_wbc/Task.h"
#include "legged_wbc/Types.h"
#include "legged_wbc/ModelHelperFunctions.h"
#include "legged_wbc/Rotation.hpp"
#include "legged_wbc/RotationDerivativesTransforms.h"
#include "pinocchio/spatial/explog.hpp"
#include "legged_wbc/WbcBase.h"

#include <logger/CsvLogger.h>

#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/math/rpy.hpp>

namespace legged {

vector_t WbcBase::update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired, 
                         const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                         scalar_t /*period*/ , std::string /*method*/) {
  contactFlag_ = contactFlag;
  numContacts_ = std::accumulate(contactFlag_.begin(), contactFlag_.end(), 0);

  qDesired_ = qDesired;
  vDesired_ = vDesired;
  fDesired_ = fDesired;
  qMeasured_ = qMeasured;
  vMeasured_ = vMeasured;
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
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, leggedModel_.nDof());
    pinocchio::getFrameJacobian(model, data, leggedModel_.contact3DofIds()[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
    jMeasured_.block(3 * i, 0, 3, leggedModel_.nDof()) = jac.template topRows<3>();
  }

  // SwingLegTask & NoContactMotionTask
  pinocchio::computeJointJacobiansTimeVariation(model, data, qMeasured_, vMeasured_);
  djMeasured_ = matrix_t(3 * leggedModel_.nContacts3Dof(), leggedModel_.nDof());
  for (size_t i = 0; i < leggedModel_.nContacts3Dof(); ++i) {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, leggedModel_.nDof());
    pinocchio::getFrameJacobianTimeVariation(model, data, leggedModel_.contact3DofIds()[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
    djMeasured_.block(3 * i, 0, 3, leggedModel_.nDof()) = jac.template topRows<3>();
  }

  if(verbose_) {
    std::cout << "[WbcBase] MMeasured:\n" << MMeasured_ << std::endl;
    std::cout << "[WbcBase] nleMeasured:" << nleMeasured_.transpose() << std::endl;
    std::cout << "[WbcBase] jMeasured:\n" << jMeasured_ << std::endl;
    std::cout << "[WbcBase] djMeasured:\n" << djMeasured_ << std::endl;
  }
}

void WbcBase::updateDesired() {
  const auto& model = leggedModel_.model();
  auto& data = leggedModel_.data();

  comDesired_ = pinocchio::centerOfMass(model, data, qDesired_);

  ADesired_ = matrix_t(6, leggedModel_.nDof());
  dADesired_ = matrix_t(6, leggedModel_.nDof());
  ADesired_ = pinocchio::computeCentroidalMap(model, data, qDesired_);
  dADesired_ = pinocchio::dccrba(model, data, qDesired_, vDesired_);

  if(verbose_) {
    std::cout << "[WbcBase] ADesired:\n" << ADesired_ << std::endl;
    std::cout << "[WbcBase] dADesired:\n" << dADesired_ << std::endl;
  }
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

Task WbcBase::formulateBaseAccelTask(scalar_t period) {
  matrix_t a(6, numDecisionVars_);
  a.setZero();
  a.block(0, 0, 6, 6) = matrix_t::Identity(6, 6);

  vector_t jointAccel = (vDesired_ - vDesiredLast_).tail(leggedModel_.nJoints());
  vDesiredLast_ = vDesired_;

  const Matrix6 Ab = ADesired_.template leftCols<6>();
  const auto AbInv = computeFloatingBaseCentroidalMomentumMatrixInverse(Ab);
  const auto Aj = ADesired_.rightCols(leggedModel_.nJoints());

  Vector6 centroidalMomentumRate = mass_ * getNormalizedCentroidalMomentumRate(mass_, 
                                                                              comDesired_,
                                                                              leggedModel_.contact3DofPoss(qDesired_),
                                                                              leggedModel_.contact6DofPoss(qDesired_),
                                                                              fDesired_);
  centroidalMomentumRate.noalias() -= dADesired_ * vDesired_;
  centroidalMomentumRate.noalias() -= Aj * jointAccel;

  Vector6 b = AbInv * centroidalMomentumRate;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] BaseAccelTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }

  return {a, b, matrix_t(), vector_t()};
}

Task WbcBase::formulateBaseAccelTaskPD(scalar_t period) {
  matrix_t a(6, numDecisionVars_);
  a.setZero();
  a.block(0, 0, 6, 6) = matrix_t::Identity(6, 6);

  vector_t jointAccel = (vDesired_ - vDesiredLast_).tail(leggedModel_.nJoints());
  vDesiredLast_ = vDesired_;

  Vector6 pos_error, vel_error, accel, b; 

  // https://github.com/stack-of-tasks/pinocchio/issues/16 pinocchio store quat in [x,y,w,z]
  Eigen::Vector4d quat_des = quat_wxyz(qDesired_.segment(3,4));
  Eigen::Vector4d quat = quat_wxyz(qMeasured_.segment(3,4));
  pos_error << qDesired_.head(3) - qMeasured_.head(3),
                quat_boxminusL(quat_des, quat);

  // Representation-Free Model Predictive Control for Dynamic Motions in Quadrupeds (https://arxiv.org/pdf/2012.10002 p5 equ-29,30) 
  Eigen::Vector3d w_des = vDesired_.segment(3,3);
  Eigen::Vector3d w = vMeasured_.segment(3,3);
  Eigen::Matrix3d R_des = quat_ToR(quat_des);
  Eigen::Matrix3d R = quat_ToR(quat);
  vel_error << vDesired_.head(3) - vMeasured_.head(3),
               R.transpose()*R_des*w_des - w; 
  
  b = baseAccelKp_.asDiagonal() * pos_error + baseAccelKd_.asDiagonal() * vel_error;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] BaseAccelTaskPD " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
      
  baseAccTask_ = Task(a, b, matrix_t(), vector_t());
  return baseAccTask_;
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
      Eigen::Vector3d accel = swingKp_ * (posDesired[i] - posMeasured[i]) + swingKd_ * (velDesired[i] - velMeasured[i]);
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

// 将 YAML list 转换为 Eigen::VectorXd
inline Eigen::VectorXd yamlToEigenVector(const YAML::Node& node) {
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("YAML node is not a valid sequence.");
    }
    std::vector<double> vec = node.as<std::vector<double>>();
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}

void WbcBase::loadTasksSetting(const std::string& configFile) {
  std::cout << "[WbcBase] Load config from " << configFile << std::endl;
  YAML::Node configNode = YAML::LoadFile(configFile);

  verbose_ = configNode["verbose"].as<bool>();

  leggedModel_.loadUrdf(configNode["urdfPath"].as<std::string>(), "quaternion",
                       configNode["baseName"].as<std::string>(), 
                       configNode["contact3DofNames"].as<std::vector<std::string>>(), 
                       configNode["contact6DofNames"].as<std::vector<std::string>>(), verbose_);

  mass_ = pinocchio::computeTotalMass(leggedModel_.model());

  numDecisionVars_ = leggedModel_.nDof() + 3 * leggedModel_.nContacts3Dof() + leggedModel_.nContacts6Dof()*6 + leggedModel_.nJoints();
  qMeasured_ = vector_t(leggedModel_.nqBase());
  vMeasured_ = vector_t(leggedModel_.nDof());
  qDesired_ = vector_t(leggedModel_.nqBase());
  vDesired_ = vector_t(leggedModel_.nDof());
  vDesiredLast_ = vector_t(leggedModel_.nDof());
  fDesired_ = vector_t(leggedModel_.nContacts3Dof()*3 + leggedModel_.nContacts6Dof()*6);

  baseAccelKp_ = yamlToEigenVector(configNode["baseAccelTask"]["baseAcc_kp"]);
  baseAccelKd_ = yamlToEigenVector(configNode["baseAccelTask"]["baseAcc_kd"]);
  swingKp_ = configNode["swingLegTask"]["kp"].as<double>();
  swingKd_ = configNode["swingLegTask"]["kd"].as<double>();
  torqueLimits_ = yamlToEigenVector(configNode["torqueLimitsTask"]);
  frictionCoeff_ = configNode["frictionConeTask"]["frictionCoefficient"].as<double>();

  jointKp_ = configNode["jointKp"].as<double>();
  jointKd_ = configNode["jointKd"].as<double>();

  if(true) {
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    std::cout << "[WbcBase] mass: " << mass_ << std::endl;
    std::cout << "[WbcBase] numDecisionVars: " << numDecisionVars_ << std::endl;
    std::cout << "[WbcBase] size of qMeasured : " << qMeasured_.size() << std::endl;
    std::cout << "[WbcBase] size of fDesired_: " << fDesired_.size() << std::endl;

    std::cout << "[WbcBase] baseAccelKp: " << baseAccelKp_.transpose() << std::endl;
    std::cout << "[WbcBase] baseAccelKd: " << baseAccelKd_.transpose() << std::endl;
    std::cout << "[WbcBase] swingKp: " << swingKp_ << std::endl;
    std::cout << "[WbcBase] swingKd: " << swingKd_ << std::endl;
    std::cout << "[WbcBase] torqueLimits: " << torqueLimits_.transpose() << std::endl;
    std::cout << "[WbcBase] frictionCoeff: " << frictionCoeff_ << std::endl;

    std::cout << "[WbcBase] jointKp: " << jointKp_ << std::endl;
    std::cout << "[WbcBase] jointKd: " << jointKd_ << std::endl;
  }
}

void WbcBase::log(const vector_t& x){
    CsvLogger& logger = CsvLogger::getInstance();
}

}  // namespace legged
