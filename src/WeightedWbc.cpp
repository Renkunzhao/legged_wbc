//
// Created by qiayuan on 22-12-23.
//

#include "legged_wbc/WeightedWbc.h"
#include "legged_wbc/WbcBase.h"
#include "logger/CsvLogger.h"

#include <iostream>
#include <yaml-cpp/yaml.h>
#include <qpOASES.hpp>

namespace legged {

vector_t WeightedWbc::update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired,
                             const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                             scalar_t period, std::string method) {
  WbcBase::update(qDesired, vDesired, fDesired, qMeasured, vMeasured, contactFlag, period);

  // Constraints
  Task constraints = formulateConstraints();
  size_t numConstraints = constraints.b_.size() + constraints.f_.size();

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(numConstraints, getNumDecisionVars());
  vector_t lbA(numConstraints), ubA(numConstraints);  // clang-format off
  A << constraints.a_,
       constraints.d_;

  lbA << constraints.b_,
         -qpOASES::INFTY * vector_t::Ones(constraints.f_.size());
  ubA << constraints.b_,
         constraints.f_;  // clang-format on

  Task weighedTask;
  // Cost
  if (method == "centroidal") {
    // Centroidal method
    weighedTask = formulateWeightedTasks(period, "centroidal");
  } else if (method == "pd") {
    // PD method
    weighedTask = formulateWeightedTasks(period, "pd");
  } else {
    return vector_t::Zero(getNumDecisionVars());
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H = weighedTask.a_.transpose() * weighedTask.a_;
  vector_t g = -weighedTask.a_.transpose() * weighedTask.b_;

  // Solve
  auto qpProblem = qpOASES::QProblem(getNumDecisionVars(), numConstraints);
  qpOASES::Options options;
  options.setToMPC();
  options.printLevel = qpOASES::PL_LOW;
  options.enableEqualities = qpOASES::BT_TRUE;
  qpProblem.setOptions(options);
  int nWsr = 20;

  int status = qpProblem.init(H.data(), g.data(), A.data(), nullptr, nullptr, lbA.data(), ubA.data(), nWsr);
  if (status != qpOASES::SUCCESSFUL_RETURN) {
    // std::cout << "QP failed to solve, status=" << status << std::endl;
  }

  vector_t qpSol(getNumDecisionVars());
  qpProblem.getPrimalSolution(qpSol.data());

  // auto objVal = qpProblem.getObjVal();
  // double wbcCost = 0.5*qpSol.dot(H*qpSol) + g.dot(qpSol);

  CsvLogger& logger = CsvLogger::getInstance();
  logger.update("qpStatus", status);
  logger.update("qpSol", qpSol);
  // logger.update("objVal", objVal);
  // logger.update("WbcCost", wbcCost);
  return qpSol;
}

Task WeightedWbc::formulateConstraints() {
  return formulateFloatingBaseEomTask() + formulateTorqueLimitsTask() + formulateFrictionConeTask() + formulateNoContactMotionTask();
}

Task WeightedWbc::formulateWeightedTasks(scalar_t period, std::string method) {
  if (method == "centroidal") {
    return formulateSwingLegTask() * weightSwingLeg_ + formulateBaseAccelTask(period) * weightBaseAccel_ +
          formulateContactForceTask() * weightContactForce_;
  } else if (method == "pd") {
    return formulateSwingLegTask() * weightSwingLeg_ + formulateBaseAccelTaskPD(period) * weightBaseAccel_ +
      formulateContactForceTask() * weightContactForce_ + formulateJointTorqueTask() * weightJointTorque_;
  }
}

void WeightedWbc::loadTasksSetting(const std::string& configFile) {
  WbcBase::loadTasksSetting(configFile);

  std::cout << "[WeightedWbc] Load config from " << configFile << std::endl;
  YAML::Node configNode = YAML::LoadFile(configFile);

  weightBaseAccel_ = configNode["weight"]["baseAccel"].as<double>();
  weightContactForce_ = configNode["weight"]["contactForce"].as<double>();
  weightSwingLeg_ = configNode["weight"]["swingLeg"].as<double>();
  weightJointTorque_ = configNode["weight"]["jointTorque"].as<double>();

  if(verbose_) {
    std::cout << "[WeightedWbc] weightBaseAccel: " << weightBaseAccel_ << std::endl;
    std::cout << "[WeightedWbc] weightContactForce: " << weightContactForce_ << std::endl;
    std::cout << "[WeightedWbc] weightSwingLeg: " << weightSwingLeg_ << std::endl;
    std::cout << "[WeightedWbc] weightJointTorque: " << weightJointTorque_ << std::endl;
  }
}

void WeightedWbc::log(const vector_t& x){
    WbcBase::log(x);
    CsvLogger& logger = CsvLogger::getInstance();

    logger.update("swingLegCost", computeCost(swingLegTask_, x, weightSwingLeg_));
    logger.update("baseAccCost", computeCost(baseAccTask_, x, weightBaseAccel_));
    logger.update("contactForceCost", computeCost(contactForceTask_, x, weightContactForce_));
    logger.update("jointTorqueCost", computeCost(jointTorqueTask_, x, weightJointTorque_));
}

}  // namespace legged
