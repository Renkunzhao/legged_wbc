//
// Created by Kunzhao on 2025/8/31.
//

#pragma once

#include "legged_wbc/Task.h"
#include "legged_wbc/LeggedModel.h"
#include "legged_wbc/Types.h"

#include <array>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace legged {

class WbcParameters {
  public:
  std::string motionName_;

  vector_t baseAccelKp_, baseAccelKd_, comAccelKp_, comAccelKd_;
  scalar_t swingKp_, swingKd_;
  scalar_t jointKp_, jointKd_;

  Eigen::VectorXd weightBaseAccel_, weightComAccel_, weightContactForce_;
  scalar_t weightSumFz_, weightSwingLeg_, weightJointTorque_;
};

// Decision Variables: x = [\dot u^T, F^T, \tau^T]^T
class WbcBase {
  using Vector6 = Eigen::Matrix<scalar_t, 6, 1>;
  using Matrix6 = Eigen::Matrix<scalar_t, 6, 6>;

 public:
  WbcBase() = default;

  virtual void loadTasksSetting(const std::string& configFile);

  virtual void log(const vector_t& x);

  virtual vector_t update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired,
                          const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                          scalar_t period, std::string method = "centroidal");

  size_t mass() const {return mass_;}
  LeggedModel& leggedModel() {return leggedModel_;}

  double getJointKp() const {return wbcParam_.jointKp_;}
  double getJointKd() const {return wbcParam_.jointKd_;}

  void setWbcParam(const std::string& motionName);

 protected:
  double inline computeCost(Task task, vector_t x, double weight = 1){
    vector_t y =  (task.a_ * x - task.b_);
    return 0.5 * weight*weight * (y.squaredNorm() - task.b_.squaredNorm());
  }
  double inline computeCost(Task task, vector_t x, vector_t weight){
    if (task.a_.rows()!=weight.size()) {
      throw runtime_error("[WbcBase] computeCost task and weight dimension mismatch.");
    }
    vector_t y = weight.asDiagonal() * (task.a_ * x - task.b_);
    vector_t b = weight.asDiagonal() * task.b_;
    return 0.5 * (y.squaredNorm() - b.squaredNorm());
  }


  void updateMeasured();
  void updateDesired();

  size_t getNumDecisionVars() const { return numDecisionVars_; }

  Task formulateFloatingBaseEomTask();
  Task formulateTorqueLimitsTask();
  Task formulateNoContactMotionTask();
  Task formulateFrictionConeTask();
  Task formulateBaseAccelTask(scalar_t period);
  Task formulateBaseAccelTaskPD();
  Task formulateComAccelTask();
  Task formulateSwingLegTask();
  Task formulateContactForceTask();
  Task formulateSumFzTask();
  Task formulateJointTorqueTask();

  LeggedModel leggedModel_;
  size_t numDecisionVars_;

  double mass_;
  vector_t qMeasured_, vMeasured_, qDesired_, vDesired_, vDesiredLast_, fDesired_;
  size_t numContacts_;
  std::array<bool, 4> contactFlag_;
  matrix_t MMeasured_, nleMeasured_, jMeasured_, djMeasured_;
  Eigen::Vector3d p_comMeasured_, v_comMeasured_, p_comDesired_;
  Vector6 hMeasured_, hDesired_;
  matrix_t AMeasured_, dAMeasured_, ADesired_, dADesired_;

  // 将 YAML list 转换为 Eigen::VectorXd
  inline Eigen::VectorXd yamlToEigenVector(const YAML::Node& node) {
      if (!node || !node.IsSequence()) {
          throw std::runtime_error("YAML node is not a valid sequence.");
      }
      std::vector<double> vec = node.as<std::vector<double>>();
      return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
  }

  // Task Parameters:
  bool verbose_;
  vector_t torqueLimits_ = vector_t::Zero(3);
  scalar_t frictionCoeff_{};

  void loadWbcParam(const std::string& motionFile, bool verbose);
  vector<WbcParameters> wbcParamList_;
  WbcParameters wbcParam_;

  // Task
  Task swingLegTask_, baseAccTask_, comAccTask_, contactForceTask_, SumFzTask_, jointTorqueTask_;
};

}  // namespace legged
