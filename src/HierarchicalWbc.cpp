//
// Created by qiayuan on 22-12-23.
//

#include "legged_wbc/HierarchicalWbc.h"

#include "legged_wbc/HoQp.h"

namespace legged {
vector_t HierarchicalWbc::update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired,
                                 const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                                 scalar_t period, std::string method) {
  WbcBase::update(qDesired, vDesired, fDesired, qMeasured, vMeasured, contactFlag, period);

  Task task0 = formulateFloatingBaseEomTask() + formulateTorqueLimitsTask() + formulateFrictionConeTask() + formulateNoContactMotionTask();
  Task task1 = formulateBaseAccelTask(period) + formulateSwingLegTask();
  Task task2 = formulateContactForceTask();
  HoQp hoQp(task2, std::make_shared<HoQp>(task1, std::make_shared<HoQp>(task0)));

  return hoQp.getSolutions();
}

}  // namespace legged
