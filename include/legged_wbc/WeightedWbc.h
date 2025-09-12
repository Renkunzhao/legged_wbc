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
  std::string qpSolver_ = "qpOASES";
  scalar_t weightSwingLeg_, weightBaseAccel_, weightContactForce_, weightJointTorque_;

  // --- solver backends 封装 ---
  vector_t solveWithQPOases(const Eigen::MatrixXd& H,
                            const vector_t& g,
                            const Eigen::MatrixXd& A,
                            const vector_t& lbA,
                            const vector_t& ubA,
                            int maxWsr = 20);

  vector_t solveWithOSQP(const Eigen::MatrixXd& H,
                         const vector_t& g,
                         const Eigen::MatrixXd& A,
                         const vector_t& lbA,
                         const vector_t& ubA,
                         double hessian_reg = 1e-9);

  static Eigen::SparseMatrix<double> toSparse(const Eigen::MatrixXd& M) {
    return M.sparseView();
  }
};

}  // namespace legged