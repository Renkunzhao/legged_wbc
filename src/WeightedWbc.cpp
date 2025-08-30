//
// Created by qiayuan on 22-12-23.
//

#include "legged_wbc/WeightedWbc.h"

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

  qpProblem.init(H.data(), g.data(), A.data(), nullptr, nullptr, lbA.data(), ubA.data(), nWsr);
  vector_t qpSol(getNumDecisionVars());

  qpProblem.getPrimalSolution(qpSol.data());
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
      formulateContactForceTask() * weightContactForce_;
  }
}

void WeightedWbc::loadTasksSetting(const std::string& configFile, bool verbose) {
  WbcBase::loadTasksSetting(configFile, verbose);

  std::cout << "[WeightedWbc]: Load config from " << configFile << std::endl;
  YAML::Node configNode = YAML::LoadFile(configFile);

  weightBaseAccel_ = configNode["weight"]["baseAccel"].as<double>();
  weightContactForce_ = configNode["weight"]["baseAccel"].as<double>();
  weightSwingLeg_ = configNode["weight"]["swingLeg"].as<double>();
  std::cout << "[WeightedWbc]: Config finished." << std::endl;
}

}  // namespace legged
