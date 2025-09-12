//
// Created by qiayuan on 22-12-23.
//

#include "legged_wbc/WbcBase.h"
#include <string>

namespace legged {

class WeightedWbc : public WbcBase {
 public:
  using WbcBase::WbcBase;

  vector_t update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired,
                  const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                  scalar_t period, std::string method="centroidal") override;

  void loadTasksSetting(const std::string& configFile) override;

  void log(const vector_t& x) override;

 protected:
  virtual Task formulateConstraints();
  virtual Task formulateWeightedTasks(scalar_t period, std::string method = "centroidal");

 private:
  scalar_t weightSwingLeg_, weightBaseAccel_, weightContactForce_, weightJointTorque_;
};

}  // namespace legged