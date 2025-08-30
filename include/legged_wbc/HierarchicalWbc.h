//
// Created by qiayuan on 22-12-23.
//
#pragma once

#include "legged_wbc/WbcBase.h"

namespace legged {

class HierarchicalWbc : public WbcBase {
 public:
  using WbcBase::WbcBase;

  vector_t update(const vector_t& qDesired, const vector_t& vDesired, const vector_t& fDesired,
                  const vector_t& qMeasured, const vector_t& vMeasured, std::array<bool, 4> contactFlag,
                  scalar_t period, std::string method) override;
};

}  // namespace legged
