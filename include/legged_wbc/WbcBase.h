//
// Created by Kunzhao on 2025/8/31.
//

#pragma once

#include "legged_wbc/LeggedState.h"
#include "legged_wbc/Task.h"
#include "legged_wbc/LeggedModel.h"
#include "legged_wbc/Types.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace legged {

class WbcParameters {
  public:
  string motionName_;

  vector<string> constraintList_;

  vector_t baseAccelKp_, baseAccelKd_, comKp_, comKd_;
  scalar_t swingKp_, swingKd_;
  scalar_t jointKp_, jointKd_;
  scalar_t stanceZKp_, stanceZKd_;

  Eigen::VectorXd weightBaseAccel_, weightCom_, weightContactForce_, weightNoContactMotion_;
  scalar_t weightSumFz_, weightSwingLeg_, weightJointTorque_, weightFootZ_ = 0.0;
};

// Decision Variables: x = [\dot u^T, F^T, \tau^T]^T
class WbcBase {
  using Vector6 = Eigen::Matrix<scalar_t, 6, 1>;
  using Matrix6 = Eigen::Matrix<scalar_t, 6, 6>;

 public:
  WbcBase() = default;

  virtual void loadTasksSetting(const std::string& configFile);

  virtual void log(const vector_t& x);

  virtual vector_t update(LeggedState des_state, LeggedState real_state,
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
  Task formulateBaseAccelTaskPD();
  Task formulateComTask();
  Task formulateSwingLegTask();
  Task formulateContactForceTask();
  Task formulateSumFzTask();
  Task formulateJointTorqueTask();

  // Soft terrain
  Task formulateNoSlipXYTask();
  Task formulateFootZTask();

  LeggedModel leggedModel_;
  size_t numDecisionVars_;

  double mass_;
  LeggedState des_state_, real_state_;
  vector_t qMeasured_, vMeasured_, qDesired_, vDesired_, vDesiredLast_, fDesired_;
  Eigen::Vector3d comDes_, vcomDes_, comAct_, vcomAct_;
  Vector6 hgDes_, hgAct_;
  size_t numContacts_;
  vector<bool> contactFlag_;
  vector<Vector3d> ee3DofPos_des_, ee3DofPos_act_, ee3DofVel_des_, ee3DofVel_act_;
  matrix_t MMeasured_, nleMeasured_, jMeasured_, djMeasured_;
  matrix_t AMeasured_, dAMeasured_;

  // Task Parameters:
  bool verbose_;
  vector_t torqueLimits_ = vector_t::Zero(3);
  scalar_t frictionCoeff_{};

  void loadWbcParam(const std::string& motionFile, bool verbose);
  vector<WbcParameters> wbcParamList_;
  WbcParameters wbcParam_;

  // Task
  Task swingLegTask_, baseAccTask_, comTask_, contactForceTask_, SumFzTask_, jointTorqueTask_, noContactMotionTask_, footZTask_;

  size_t counter_ = 0, logInterval_ = 50;
};

}  // namespace legged
