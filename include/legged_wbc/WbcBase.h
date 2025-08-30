//
// Created by qiayuan on 2022/7/1.
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
  WbcBase();

  virtual void loadTasksSetting(const std::string& configFile, bool verbose);

  virtual vector_t update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired,
                          const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                          scalar_t period, std::string method = "centroidal");

 protected:
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

  LeggedModel leggedModel;
  size_t numDecisionVars_;

  size_t mass_;
  vector_t qMeasured_, vMeasured_, qDesired_, vDesired_, vDesiredLast_, fDesired_;
  size_t numContacts_;
  std::array<bool, 4> contactFlag_;
  matrix_t MMeasured_, nleMeasured_, jMeasured_, djMeasured_;
  Eigen::Vector3d comDesired_;
  matrix_t ADesired_, dADesired_;

  // Task Parameters:
  vector_t torqueLimits_ = vector_t::Zero(3), baseAccelKp_ = vector_t::Zero(6), baseAccelKd_ = vector_t::Zero(6);
  scalar_t frictionCoeff_{}, swingKp_{}, swingKd_{};
};

}  // namespace legged
