//
// Created by qiayuan on 2022/7/1.
//
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "legged_wbc/Types.h"
#include "legged_wbc/ModelHelperFunctions.h"
#include "legged_wbc/RotationDerivativesTransforms.h"
#include "legged_wbc/WbcBase.h"

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
  const auto& model = leggedModel.model();
  auto& data = leggedModel.data();

  // EOM Task
  MMeasured_ = matrix_t(leggedModel.nDof(), leggedModel.nDof());
  nleMeasured_ = vector_t(leggedModel.nDof());
  MMeasured_ = pinocchio::crba(model, data, qMeasured_);
  MMeasured_.triangularView<Eigen::StrictlyLower>() = MMeasured_.transpose().triangularView<Eigen::StrictlyLower>();
  nleMeasured_ = pinocchio::nonLinearEffects(model, data, qMeasured_, vMeasured_);

  pinocchio::forwardKinematics(model, data, qMeasured_, vMeasured_);

  // EOM Task & SwingLegTask & NoContactMotionTask
  pinocchio::computeJointJacobians(model, data);
  jMeasured_ = matrix_t(3 * leggedModel.nContacts3Dof(), leggedModel.nDof());
  for (size_t i = 0; i < leggedModel.nContacts3Dof(); ++i) {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, leggedModel.nDof());
    pinocchio::getFrameJacobian(model, data, leggedModel.contact3DofIds()[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
    jMeasured_.block(3 * i, 0, 3, leggedModel.nDof()) = jac.template topRows<3>();
  }

  // SwingLegTask & NoContactMotionTask
  pinocchio::computeJointJacobiansTimeVariation(model, data, qMeasured_, vMeasured_);
  djMeasured_ = matrix_t(3 * leggedModel.nContacts3Dof(), leggedModel.nDof());
  for (size_t i = 0; i < leggedModel.nContacts3Dof(); ++i) {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, leggedModel.nDof());
    pinocchio::getFrameJacobianTimeVariation(model, data, leggedModel.contact3DofIds()[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
    djMeasured_.block(3 * i, 0, 3, leggedModel.nDof()) = jac.template topRows<3>();
  }

  if(verbose_) {
    std::cout << "[WbcBase] MMeasured:\n" << MMeasured_ << std::endl;
    std::cout << "[WbcBase] nleMeasured:" << nleMeasured_.transpose() << std::endl;
    std::cout << "[WbcBase] jMeasured:\n" << jMeasured_ << std::endl;
    std::cout << "[WbcBase] djMeasured:\n" << djMeasured_ << std::endl;
  }
}

void WbcBase::updateDesired() {
  const auto& model = leggedModel.model();
  auto& data = leggedModel.data();

  comDesired_ = pinocchio::centerOfMass(model, data, qDesired_);

  ADesired_ = matrix_t(6, leggedModel.nDof());
  dADesired_ = matrix_t(6, leggedModel.nDof());
  ADesired_ = pinocchio::computeCentroidalMap(model, data, qDesired_);
  dADesired_ = pinocchio::dccrba(model, data, qDesired_, vDesired_);

  if(verbose_) {
    std::cout << "[WbcBase] ADesired:\n" << ADesired_ << std::endl;
    std::cout << "[WbcBase] dADesired:\n" << dADesired_ << std::endl;
  }
}

Task WbcBase::formulateFloatingBaseEomTask() {
  matrix_t s(leggedModel.nJoints(), leggedModel.nDof());
  s.block(0, 0, leggedModel.nJoints(), 6).setZero();
  s.block(0, 6, leggedModel.nJoints(), leggedModel.nJoints()).setIdentity();

  matrix_t a = (matrix_t(leggedModel.nDof(), numDecisionVars_) << MMeasured_, -jMeasured_.transpose(), -s.transpose()).finished();
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
  matrix_t d(2 * leggedModel.nJoints(), numDecisionVars_);
  d.setZero();
  matrix_t i = matrix_t::Identity(leggedModel.nJoints(), leggedModel.nJoints());
  d.block(0, leggedModel.nDof() + 3 * leggedModel.nContacts3Dof(), leggedModel.nJoints(), leggedModel.nJoints()) = i;
  d.block(leggedModel.nJoints(), leggedModel.nDof() + 3 * leggedModel.nContacts3Dof(), leggedModel.nJoints(),
          leggedModel.nJoints()) = -i;
  vector_t f(2 * leggedModel.nJoints());
  for (size_t l = 0; l < 2 * leggedModel.nJoints() / 3; ++l) {
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
  for (size_t i = 0; i < leggedModel.nContacts3Dof(); i++) {
    if (contactFlag_[i]) {
      a.block(3 * j, 0, 3, leggedModel.nDof()) = jMeasured_.block(3 * i, 0, 3, leggedModel.nDof());
      b.segment(3 * j, 3) = -djMeasured_.block(3 * i, 0, 3, leggedModel.nDof()) * vMeasured_;
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
  matrix_t a(3 * (leggedModel.nContacts3Dof() - numContacts_), numDecisionVars_);
  a.setZero();
  size_t j = 0;
  for (size_t i = 0; i < leggedModel.nContacts3Dof(); ++i) {
    if (!contactFlag_[i]) {
      a.block(3 * j++, leggedModel.nDof() + 3 * i, 3, 3) = matrix_t::Identity(3, 3);
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
  for (size_t i = 0; i < leggedModel.nContacts3Dof(); ++i) {
    if (contactFlag_[i]) {
      d.block(5 * j++, leggedModel.nDof() + 3 * i, 5, 3) = frictionPyramic;
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

  vector_t jointAccel = (vDesired_ - vDesiredLast_).tail(leggedModel.nJoints());
  vDesiredLast_ = vDesired_;

  const Matrix6 Ab = ADesired_.template leftCols<6>();
  const auto AbInv = computeFloatingBaseCentroidalMomentumMatrixInverse(Ab);
  const auto Aj = ADesired_.rightCols(leggedModel.nJoints());

  Vector6 centroidalMomentumRate = mass_ * getNormalizedCentroidalMomentumRate(mass_, 
                                                                              comDesired_,
                                                                              leggedModel.contact3DofPoss(qDesired_),
                                                                              leggedModel.contact6DofPoss(qDesired_),
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

  vector_t jointAccel = (vDesired_ - vDesiredLast_).tail(leggedModel.nJoints());
  vDesiredLast_ = vDesired_;

  Vector6 pos_error, vel_error, accel, b; 

  Eigen::Vector3d eulerZYX_des = qDesired_.segment<3>(3);
  Eigen::Vector3d eulerZYX = qMeasured_.segment<3>(3);

  Eigen::Matrix3d R_des = pinocchio::rpy::rpyToMatrix(eulerZYX_des.reverse());
  Eigen::Matrix3d R = pinocchio::rpy::rpyToMatrix(eulerZYX.reverse());

  pinocchio::SE3 T_des(R_des, qDesired_.head<3>());
  pinocchio::SE3 T(R, qMeasured_.head<3>());

  pos_error = pinocchio::log6(T_des * T.inverse()).toVector();

  Eigen::Vector3d eulerZYX_dot_des = vDesired_.segment<3>(3);
  Eigen::Vector3d eulerZYX_dot = vMeasured_.segment<3>(3);
  
  Eigen::Vector3d angularVel_des = getGlobalAngularVelocityFromEulerAnglesZyxDerivatives(eulerZYX_des, eulerZYX_dot_des);
  Eigen::Vector3d angularVel = getGlobalAngularVelocityFromEulerAnglesZyxDerivatives(eulerZYX, eulerZYX_dot);

  vel_error << vDesired_.head<3>() - vMeasured_.head<3>(), 
               angularVel_des - angularVel;
  
  accel = baseAccelKp_.asDiagonal() * pos_error + baseAccelKd_.asDiagonal() * vel_error;

  b << accel.head<3>(), 
      getEulerAnglesZyxDerivativesFromGlobalAngularAcceleration(eulerZYX, eulerZYX_dot, accel.segment<3>(3).eval());

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] BaseAccelTaskPD " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
      
  return {a, b, matrix_t(), vector_t()};
}

Task WbcBase::formulateSwingLegTask() {
  std::vector<Eigen::Vector3d> posMeasured = leggedModel.contact3DofPoss(qMeasured_);
  std::vector<Eigen::Vector3d> velMeasured = leggedModel.contact3DofVels(qMeasured_, vMeasured_);
  std::vector<Eigen::Vector3d> posDesired = leggedModel.contact3DofPoss(qDesired_);
  std::vector<Eigen::Vector3d> velDesired = leggedModel.contact3DofVels(qDesired_, vDesired_);

  matrix_t a(3 * (leggedModel.nContacts3Dof() - numContacts_), numDecisionVars_);
  vector_t b(a.rows());
  a.setZero();
  b.setZero();
  size_t j = 0;
  for (size_t i = 0; i < leggedModel.nContacts3Dof(); ++i) {
    if (!contactFlag_[i]) {
      Eigen::Vector3d accel = swingKp_ * (posDesired[i] - posMeasured[i]) + swingKd_ * (velDesired[i] - velMeasured[i]);
      a.block(3 * j, 0, 3, leggedModel.nDof()) = jMeasured_.block(3 * i, 0, 3, leggedModel.nDof());
      b.segment(3 * j, 3) = accel - djMeasured_.block(3 * i, 0, 3, leggedModel.nDof()) * vMeasured_;
      j++;
    }
  }

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] SwingLegTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  return {a, b, matrix_t(), vector_t()};
}

Task WbcBase::formulateContactForceTask() {
  matrix_t a(3 * leggedModel.nContacts3Dof(), numDecisionVars_);
  vector_t b(a.rows());
  a.setZero();

  for (size_t i = 0; i < leggedModel.nContacts3Dof(); ++i) {
    a.block(3 * i, leggedModel.nDof() + 3 * i, 3, 3) = matrix_t::Identity(3, 3);
  }
  b = fDesired_;

  if(verbose_) {
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[WbcBase] ContactForceTask " << std::endl;
    std::cout << "[WbcBase] a:\n" << a << std::endl;
    std::cout << "[WbcBase] b: " << b.transpose() << std::endl;
  }
  
  return {a, b, matrix_t(), vector_t()};
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

  leggedModel.loadUrdf(configNode["urdfPath"].as<std::string>(), "eulerZYX",
                       configNode["baseName"].as<std::string>(), 
                       configNode["contact3DofNames"].as<std::vector<std::string>>(), 
                       configNode["contact6DofNames"].as<std::vector<std::string>>());

  verbose_ = configNode["verbose"].as<bool>();

  mass_ = pinocchio::computeTotalMass(leggedModel.model());

  numDecisionVars_ = leggedModel.nDof() + 3 * leggedModel.nContacts3Dof() + leggedModel.nJoints();
  qMeasured_ = vector_t(leggedModel.nDof());
  vMeasured_ = vector_t(leggedModel.nDof());
  qDesired_ = vector_t(leggedModel.nDof());
  vDesired_ = vector_t(leggedModel.nDof());
  vDesiredLast_ = vector_t(leggedModel.nDof());
  fDesired_ = vector_t(12);

  baseAccelKp_ = yamlToEigenVector(configNode["baseAccelTask"]["baseAcc_kp"]);
  baseAccelKd_ = yamlToEigenVector(configNode["baseAccelTask"]["baseAcc_kd"]);
  swingKp_ = configNode["swingLegTask"]["kp"].as<double>();
  swingKd_ = configNode["swingLegTask"]["kd"].as<double>();
  torqueLimits_ = yamlToEigenVector(configNode["torqueLimitsTask"]);
  frictionCoeff_ = configNode["frictionConeTask"]["frictionCoefficient"].as<double>();

  if(verbose_) {
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    std::cout << "[WbcBase] baseAccelKp: " << baseAccelKp_.transpose() << std::endl;
    std::cout << "[WbcBase] baseAccelKd: " << baseAccelKd_.transpose() << std::endl;
    std::cout << "[WbcBase] swingKp: " << swingKp_ << std::endl;
    std::cout << "[WbcBase] swingKd: " << swingKd_ << std::endl;
    std::cout << "[WbcBase] torqueLimits: " << torqueLimits_.transpose() << std::endl;
    std::cout << "[WbcBase] frictionCoeff: " << frictionCoeff_ << std::endl;
  }
}

}  // namespace legged
