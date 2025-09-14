//
// Created by Kunzhao on 2025/8/31.
//

#pragma once

#include "legged_wbc/Task.h"
#include "legged_wbc/LeggedModel.h"
#include "legged_wbc/Types.h"

#include <array>

namespace legged {

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

  void computeCost(const vector_t& x, vector_t& swingLegCost, vector_t& baseAccCost, vector_t& contactForceCost);

  double getJointKp() const {return jointKp_;}
  double getJointKd() const {return jointKd_;}
  const Task& getSwingLegTask() const {return swingLegTask_;}
  const Task& getBaseAccTask() const {return baseAccTask_;}
  const Task& getContactForceTask() const {return contactForceTask_;}

 protected:
  double inline computeCost(Task task, vector_t x, double weight = 1){
    vector_t y = task.a_ * x - task.b_;
    return 0.5 * weight * weight * (y.squaredNorm() - task.b_.squaredNorm());
  }

  void updateMeasured();
  void updateDesired();

  size_t getNumDecisionVars() const { return numDecisionVars_; }

  Task formulateFloatingBaseEomTask();
  Task formulateTorqueLimitsTask();
  Task formulateNoContactMotionTask();
  Task formulateFrictionConeTask();
  Task formulateBaseAccelTask(scalar_t period);
  Task formulateBaseAccelTaskPD(scalar_t period);
  Task formulateSwingLegTask();
  Task formulateContactForceTask();
  Task formulateJointTorqueTask();

  LeggedModel leggedModel_;
  size_t numDecisionVars_;

  double mass_;
  vector_t qMeasured_, vMeasured_, qDesired_, vDesired_, vDesiredLast_, fDesired_;
  size_t numContacts_;
  std::array<bool, 4> contactFlag_;
  matrix_t MMeasured_, nleMeasured_, jMeasured_, djMeasured_;
  Eigen::Vector3d comDesired_;
  matrix_t ADesired_, dADesired_;

  // Task Parameters:
  bool verbose_;
  vector_t torqueLimits_ = vector_t::Zero(3), baseAccelKp_ = vector_t::Zero(6), baseAccelKd_ = vector_t::Zero(6);
  scalar_t frictionCoeff_{}, swingKp_{}, swingKd_{};
  scalar_t jointKp_, jointKd_;

  // Task
  Task swingLegTask_, baseAccTask_, contactForceTask_, jointTorqueTask_;
};

}  // namespace legged
