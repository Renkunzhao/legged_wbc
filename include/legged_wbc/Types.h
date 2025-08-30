//
// Created by Kunzhao Ren on 2025/8/18.
//
// Ref: https://github.com/qiayuanl/legged_control.git
//

#pragma once

#include <Eigen/Dense>

namespace legged {

using scalar_t = double;
/** Dynamic-size vector type. */
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
/** Dynamic-size matrix type. */
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

using Vector6 = Eigen::Matrix<scalar_t, 6, 1>;
using Matrix6 = Eigen::Matrix<scalar_t, 6, 6>;

}  // namespace legged
