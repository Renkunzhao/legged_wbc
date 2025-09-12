#include "legged_wbc/WeightedWbc.h"
#include "legged_wbc/WbcBase.h"
#include "logger/CsvLogger.h"

#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <qpOASES.hpp>
#include <OsqpEigen/OsqpEigen.h>

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

  // --- 选择解算器后端 ---
  vector_t qpSol;
  if (qpSolver_ == "osqp") {
    qpSol = solveWithOSQP(H, g, A, lbA, ubA, /*hessian_reg=*/1e-9);
  } else if (qpSolver_ == "qpOASES") { // 默认 qpoases
    qpSol = solveWithQPOases(H, g, A, lbA, ubA, /*maxWsr=*/20);
  }

  // --- 日志 ---
  CsvLogger& logger = CsvLogger::getInstance();
  logger.update("qpSol", qpSol);

  return qpSol;
}

// ------------------------ qpOASES 后端 ------------------------
vector_t WeightedWbc::solveWithQPOases(const Eigen::MatrixXd& H,
                                       const vector_t& g,
                                       const Eigen::MatrixXd& A,
                                       const vector_t& lbA,
                                       const vector_t& ubA,
                                       int maxWsr) {
  auto qpProblem = qpOASES::QProblem(getNumDecisionVars(), static_cast<int>(A.rows()));
  qpOASES::Options options;
  options.setToMPC();
  options.printLevel = qpOASES::PL_LOW;
  options.enableEqualities = qpOASES::BT_TRUE;
  qpProblem.setOptions(options);

  int nWsr = maxWsr;
  int status = qpProblem.init(H.data(), g.data(), A.data(), /*lb*/nullptr, /*ub*/nullptr,
                              lbA.data(), ubA.data(), nWsr);

  vector_t qpSol(getNumDecisionVars());
  if (status == qpOASES::SUCCESSFUL_RETURN) {
    qpProblem.getPrimalSolution(qpSol.data());
  } else {
    qpSol.setZero();
  }

  CsvLogger::getInstance().update("qpStatus", status);
  return qpSol;
}

// ------------------------ OSQP (OsqpEigen) 后端 ------------------------
vector_t WeightedWbc::solveWithOSQP(const Eigen::MatrixXd& H_in,
                                    const vector_t& g_in,
                                    const Eigen::MatrixXd& A_in,
                                    const vector_t& lbA,
                                    const vector_t& ubA,
                                    double hessian_reg) {
  const int n = static_cast<int>(H_in.cols());
  const int m = static_cast<int>(A_in.rows());

  // OSQP 需要半正定 Hessian；加一点对角正则提高数值稳健性
  Eigen::MatrixXd H = H_in;
  H.diagonal().array() += hessian_reg;

  // 稀疏形式
  Eigen::SparseMatrix<double> Hs = toSparse(0.5 * (H + H.transpose())); // 强制对称
  Eigen::SparseMatrix<double> As = toSparse(A_in);
  Eigen::VectorXd g = g_in; // 直接拷贝

  OsqpEigen::Solver solver;
  solver.settings()->setWarmStart(true);
  solver.settings()->setVerbosity(false);     // 需要调试时改 true
  solver.settings()->setPolish(true);
  solver.settings()->setAlpha(1.6);
  solver.settings()->setMaxIteration(4000);

  solver.data()->setNumberOfVariables(n);
  solver.data()->setNumberOfConstraints(m);

  if (!solver.data()->setHessianMatrix(Hs)) {
    CsvLogger::getInstance().update("qpStatus", -10);
    return vector_t::Zero(n);
  }
  if (!solver.data()->setGradient(g)) {
    CsvLogger::getInstance().update("qpStatus", -11);
    return vector_t::Zero(n);
  }
  if (!solver.data()->setLinearConstraintsMatrix(As)) {
    CsvLogger::getInstance().update("qpStatus", -12);
    return vector_t::Zero(n);
  }
  Eigen::VectorXd lbEval = lbA;
  Eigen::VectorXd ubEval = ubA;
  if (!solver.data()->setLowerBound(lbEval) || !solver.data()->setUpperBound(ubEval)) {
    CsvLogger::getInstance().update("qpStatus", -13);
    return vector_t::Zero(n);
  }

  if (!solver.initSolver()) {
    CsvLogger::getInstance().update("qpStatus", -20);
    return vector_t::Zero(n);
  }

  auto status = solver.solveProblem();
  CsvLogger::getInstance().update("qpStatus", static_cast<int>(status));

  if (status != OsqpEigen::ErrorExitFlag::NoError) {
    return vector_t::Zero(n);
  }

  Eigen::VectorXd sol = solver.getSolution();
  vector_t qpSol = sol;
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

  qpSolver_ = configNode["qpSolver"].as<std::string>();
  weightBaseAccel_ = configNode["weight"]["baseAccel"].as<double>();
  weightContactForce_ = configNode["weight"]["contactForce"].as<double>();
  weightSwingLeg_ = configNode["weight"]["swingLeg"].as<double>();
  weightJointTorque_ = configNode["weight"]["jointTorque"].as<double>();

  if(true) {
    std::cout << "[WeightedWbc] qpSolver: " << qpSolver_ << std::endl;
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
