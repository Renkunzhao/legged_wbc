#pragma once
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

using namespace Eigen;

inline double smoothstep(double t, size_t order = 5) {
    t = std::clamp(t, 0.0, 1.0);

    switch (order)
    {
        case 1: // Linear
            return t;

        case 3: // Cubic smoothstep (C¹ continuous)
            return t * t * (3.0 - 2.0 * t);  // 3t² - 2t³

        case 5: // Quintic smootherstep (C² continuous)
        {
            double t2 = t * t;
            double t3 = t2 * t;
            return t3 * (10.0 + t * (6.0 * t - 15.0));  // 6t⁵ -15t⁴ +10t³
        }

        default:
            throw std::invalid_argument("smoothstep: order must be 1, 3, or 5");
    }
}

inline double lerp(double t, const double& x0, const double& x1, size_t order = 5) {
    double s = smoothstep(t, order);
    return x0 + (x1 - x0) * s;
}

// Linear interpolate: t in [0,1]
template<class DerivedA, class DerivedB>
inline typename DerivedA::PlainObject
lerp(double t,
     const Eigen::MatrixBase<DerivedA>& x0,
     const Eigen::MatrixBase<DerivedB>& x1,
     size_t order = 5) {
    return (x0 + (x1 - x0) * smoothstep(t, order)).eval();  // return a plain (materialized) vector
}

// [xyzw]
inline VectorXd baseSlerp(double t,
                        const Eigen::VectorXd& qBaseDes,
                        const Eigen::VectorXd& qBaseInit, 
                        size_t order = 5) {
    // 1. Position interpolation
    Eigen::Vector3d base_pos = lerp(t, qBaseInit.head<3>(), qBaseDes.head<3>(), order).eval();

    // 2. Orientation interpolation
    Eigen::Quaterniond quat_init(qBaseInit[6], qBaseInit[3], qBaseInit[4], qBaseInit[5]); // w,x,y,z
    Eigen::Quaterniond quat_des(qBaseDes[6], qBaseDes[3], qBaseDes[4], qBaseDes[5]);     // w,x,y,z
    quat_init.normalize();
    quat_des.normalize();

    if (quat_init.dot(quat_des) < 0.0)
        quat_des.coeffs() *= -1.0;

    Eigen::Quaterniond quat_interp = quat_init.slerp(smoothstep(t, order), quat_des);
    quat_interp.normalize();

    // 3. Combine
    Eigen::VectorXd qBase(7);
    qBase << base_pos, quat_interp.vec(), quat_interp.w();
    return qBase;
}

// 将 YAML list 转换为 Eigen::VectorXd
inline Eigen::VectorXd yamlToEigenVector(const YAML::Node& node) {
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("YAML node is not a valid sequence.");
    }
    std::vector<double> vec = node.as<std::vector<double>>();
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}
