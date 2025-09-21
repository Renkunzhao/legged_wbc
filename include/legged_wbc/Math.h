#pragma once

#include <Eigen/Dense>
#include <algorithm> // for std::max

/// @brief Moore–Penrose pseudoinverse using SVD
/// @param J  Input matrix
/// @param tol  Relative tolerance (default 1e-9)
/// @return J^+  (Moore–Penrose pseudoinverse)
inline Eigen::MatrixXd pseudoInverseSVD(const Eigen::MatrixXd &J, double tol = 1e-9) {
    // Decompose J = U Σ V^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::VectorXd sigma = svd.singularValues();

    // Threshold for treating singular values as zero
    double sigma_max = sigma(0); // largest singular value
    double tolerance = tol * std::max(J.cols(), J.rows()) * sigma_max;

    // Build Σ^+
    Eigen::VectorXd sigmaInv = sigma;
    for (long i = 0; i < sigma.size(); i++) {
        sigmaInv(i) = (sigma(i) > tolerance) ? 1.0 / sigma(i) : 0.0;
    }

    // Compute pseudoinverse: J^+ = V Σ^+ U^T
    return svd.matrixV() * sigmaInv.asDiagonal() * svd.matrixU().transpose();
}

/// @brief Damped least squares pseudoinverse
/// @param J  Input matrix
/// @param lambda  Damping factor (default 1e-6)
/// @return J^+_λ  (approximate pseudoinverse)
inline Eigen::MatrixXd pseudoInverseDLS(const Eigen::MatrixXd &J, double lambda = 1e-6) {
    Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd JJt_damped = JJt + lambda * Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols());
    return J.transpose() * JJt_damped.ldlt().solve(Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols()));
}
