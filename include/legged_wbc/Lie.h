#pragma once
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

#include <Eigen/Dense>

#include <pinocchio/math/quaternion.hpp>
#include <pinocchio/math/rpy.hpp>
#include <pinocchio/math/rotation.hpp>

namespace Lie {

using namespace Eigen;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

template <typename T>
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// w_hat * v = w x v
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p7 equ-20)
inline Eigen::Matrix3d hat(Eigen::Vector3d w){
    Eigen::Matrix3d w_hat;
    double x = w[0];
    double y = w[1];
    double z = w[2];
    w_hat <<    0, -z,  y,
                z,  0, -x,
               -y,  x,  0;
    return w_hat;
}

inline Eigen::Vector3d vee(Eigen::Matrix3d w_hat){
    if (!w_hat.isApprox(-w_hat.transpose(), 1e-6)) {
        throw std::runtime_error("[Rotation] vee: input is not skew-symmetric.");
    }

    return {w_hat(2,1), w_hat(0,2), w_hat(1,0)};
}

inline Eigen::Vector4d quat_xyzw(Eigen::Vector4d q_wxyz){
    double qw = q_wxyz(0);
    Eigen::Vector3d qv = q_wxyz.tail(3);
    Eigen::Vector4d q_xyzw;
    q_xyzw << qv, qw;
    return q_xyzw;
}

inline Eigen::Vector4d quat_wxyz(Eigen::Vector4d q_xyzw){
    Eigen::Vector3d qv = q_xyzw.head(3);
    double qw = q_xyzw(3);
    Eigen::Vector4d q_wxyz;
    q_wxyz << qw, qv;
    return q_wxyz;
}

// Unless specify, all quat following store in [w, x, y, z], and w>=0
inline Eigen::Vector4d quat_wPositive(Eigen::Vector4d q){
    double w = q(0);
    return (w >= 0.0) ? q : -q;
}

// q1 \otimes q2 = [q1]L q2 = [q2]R q1
// Robot Dynamics Lecture Notes (p20 equ-2.57)
// (https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf)
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p7 equ-19)
inline Eigen::Matrix4d quat_productMatL(Eigen::Vector4d q){
    Eigen::Matrix4d QL;
    double qw = q(0);
    Eigen::Vector3d qv = q.tail(3);
    QL <<   0, -qv.transpose(),
            qv, hat(qv);
    QL += qw*Eigen::Matrix4d::Identity();
    return QL;
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p7 equ-19)
inline Eigen::Matrix4d quat_productMatR(Eigen::Vector4d q){
    Eigen::Matrix4d QR;
    double qw = q(0);
    Eigen::Vector3d qv = q.tail(3);
    QR <<   0, -qv.transpose(),
            qv,-hat(qv);
    QR += qw*Eigen::Matrix4d::Identity();
    return QR;
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p7 equ-17)
inline Eigen::Vector4d quat_product(Eigen::Vector4d q1, Eigen::Vector4d q2){
    return quat_productMatL(q1) * q2;
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p8 equ-23)
inline Eigen::Vector4d quat_conjugate(Eigen::Vector4d q){
    Eigen::Vector4d q_conjugate;
    double qw = q(0);
    Eigen::Vector3d qv = q.tail(3);
    q_conjugate << qw, -qv;
    return q_conjugate;
}

inline void quat_isNormalized(Eigen::Vector4d q){
    if(std::abs(q.norm() - 1.0) > 1e-6) {
        std::cout << "[Rotation] q: " << q.transpose() << " norm: " << q.norm() << std::endl;
        // throw std::runtime_error("[Rotation] quaternion is not normalized.");
    }
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p23 equ-107)
// x' = q * x * q'
inline Eigen::Vector3d quat_rotateVec(Eigen::Vector4d q, Eigen::Vector3d x){
    quat_isNormalized(q);
    Eigen::Vector4d x_quat;
    x_quat << 0, x;
    Eigen::Vector4d x_quat_new = quat_product(q, quat_product(x_quat, quat_conjugate(q)));
    if (abs(x_quat_new(0)) > 1e-6) {
        throw std::runtime_error("[Rotation] quat_rotateVec: new quat is not pure.");
    }
    return x_quat_new.tail(3);
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p22 equ-101)
inline Eigen::Vector4d quat_Exp(Eigen::Vector3d rotation_vec){
    if (rotation_vec.norm() < 1e-6) {
        return {1, 0, 0, 0};
    }
    double phi = rotation_vec.norm();
    Eigen::Vector3d u = rotation_vec/rotation_vec.norm();
    Eigen::Vector4d q;
    q << cos(phi/2), sin(phi/2)*u;    
    return q;
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p23 equ-105)
inline Eigen::Vector3d quat_Log(Eigen::Vector4d q){
    quat_isNormalized(q);
    double qw = q(0);
    Eigen::Vector3d qv = q.tail(3);
    if (qw == 1 || qw == -1) {
        return Eigen::Vector3d::Zero();
    }
    double phi = 2 * std::atan2(qv.norm(), qw);
    Eigen::Vector3d u = qv/qv.norm();

    if (phi > M_PI) {
        phi = 2*M_PI - phi;
        u   = -u;
    }

    return phi*u;
}

// err = q_des \boxminus_{L} q = Log(q'*q_des), err/dt should be angular vel in local frame
// q_des and q has to be normalized 
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p44 equ-192)
inline Eigen::Vector3d quat_boxminusL(Eigen::Vector4d q_des, Eigen::Vector4d q){
    quat_isNormalized(q_des);
    quat_isNormalized(q);
    Eigen::Vector4d q_conjugate = quat_conjugate(q);
    Eigen::Vector4d q_diff = quat_product(q_conjugate, q_des);
    return quat_Log(q_diff);
}

// err = q_des \boxminus_{W} q = Log(q_des*q'), err/dt should be angular vel in world frame
// q_des and q has to be normalized 
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p45 equ-195)
inline Eigen::Vector3d quat_boxminusW(Eigen::Vector4d q_des, Eigen::Vector4d q){
    quat_isNormalized(q_des);
    quat_isNormalized(q);
    Eigen::Vector4d q_conjugate = quat_conjugate(q);
    Eigen::Vector4d q_diff = quat_product(q_des, q_conjugate);
    return quat_Log(q_diff);
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p25 equ-115)
inline Eigen::Matrix3d quat_ToR(Eigen::Vector4d q){
    quat_isNormalized(q);
    double qw = q(0);
    Eigen::Vector3d qv = q.tail(3);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    return (qw*qw - qv.transpose()*qv)*I + 2*qv*qv.transpose() + 2*qw*hat(qv);
}

inline void R_isRotationMatrix(Eigen::Matrix3d R){
    Eigen::Matrix3d RRT = R * R.transpose();
    if (!RRT.isApprox(Eigen::Matrix3d::Identity(), 1e-6)) {
        std::cout << "R*R^T\n" << RRT << std::endl;
        throw std::runtime_error("[Rotation] R*R^T is not Identity.");
    } else if ( std::abs(R.determinant()-1) > 1e-6) {
        throw std::runtime_error("[Rotation] det(R) is not +1.");
    }
}

// Rodrigues rotation formula
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p18 equ-78)
inline Eigen::Matrix3d R_Exp(Eigen::Vector3d rotation_vec){
    if (rotation_vec.norm() < 1e-6) {
        return Eigen::Matrix3d::Identity();
    }
    double phi = rotation_vec.norm();
    Eigen::Vector3d u = rotation_vec/rotation_vec.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    return cos(phi)*I + sin(phi)*hat(u) + (1-cos(phi))*u*u.transpose();
}

// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p19 equ-80)
inline Eigen::Vector3d R_Log(Eigen::Matrix3d R){
    R_isRotationMatrix(R);
    double phi = acos((R.trace()-1)/2 );
    if (std::abs(phi) < 1e-6) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d u = vee(R-R.transpose()) / (2*sin(phi));
    return phi*u;
}

// err = R_des \boxminus_{L} R = Log(R^T*R_des), err/dt should be angular vel in local frame
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p44 equ-192)
inline Eigen::Vector3d R_boxminusL(Eigen::Matrix3d R_des, Eigen::Matrix3d R){
    R_isRotationMatrix(R_des);
    R_isRotationMatrix(R);
    return R_Log(R.transpose()*R_des);
}

// err = R_des \boxminus_{W} R = Log(R_des*R^T), err/dt should be angular vel in world frame
// Quaternion kinematics for the error-state Kalman filter (https://arxiv.org/pdf/1711.02508 p45 equ-195)
inline Eigen::Vector3d R_boxminusW(Eigen::Matrix3d R_des, Eigen::Matrix3d R){
    R_isRotationMatrix(R_des);
    R_isRotationMatrix(R);
    return R_Log(R_des*R.transpose());
}

// ETH - Robot Dynamics Lecture Notes (p19 equ-2.55)
// https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
inline Eigen::Vector4d R_ToQuat(Eigen::Matrix3d R){
    double& R11 = R(0,0);
    double& R12 = R(0,1);
    double& R13 = R(0,2);
    double& R21 = R(1,0);
    double& R22 = R(1,1);
    double& R23 = R(1,2);
    double& R31 = R(2,0);
    double& R32 = R(2,1);
    double& R33 = R(2,2);
    double qw = 0.5 * sqrt(R11 + R22 + R33 + 1);
    double qx = 0.5 * sgn(R32 - R23) * sqrt(R11 - R22 - R33 + 1);
    double qy = 0.5 * sgn(R13 - R31) * sqrt(R22 - R33 - R11 + 1);
    double qz = 0.5 * sgn(R21 - R12) * sqrt(R33 - R11 - R22 + 1);
    return {qw, qx, qy, qz};
}

// A micro Lie theory for state estimation in robotics (https://zhuanlan.zhihu.com/p/4741261658, p16, equ-168)
inline Eigen::Matrix4d T_Rp(Eigen::Matrix3d R, Eigen::Vector3d p){
    Eigen::Matrix4d T;
    T << R,                                     p,
         Eigen::Vector3d::Zero().transpose(),   1;
    return T;
}

inline void T_isHomoTrans(Eigen::Matrix4d T){
    Eigen::Matrix3d R = T.block(0,0,3,3);
    Eigen::Vector3d p = T.block(0,3,3,1);
    Eigen::Vector4d bottomLine = {0,0,0,1};
    R_isRotationMatrix(R);
    if ( !T.bottomRows(1).isApprox(bottomLine.transpose(), 1e-6) ) {
        throw std::runtime_error("[Rotation] T is not homogeneous transformation matrix.");
    }
}

// A micro Lie theory for state estimation in robotics (https://zhuanlan.zhihu.com/p/4741261658, p16, equ-170)
inline Eigen::Matrix4d T_inv(Eigen::Matrix4d T){
    T_isHomoTrans(T);
    Eigen::Matrix3d R = T.block(0,0,3,3);
    Eigen::Vector3d p = T.block(0,3,3,1);
    Eigen::Matrix4d T_inv;
    T_inv << R.transpose(), -R.transpose()*p,
             0,0,0,         1;
    return T_inv;
}

// xi = [pho, theta]
// A micro Lie theory for state estimation in robotics (https://zhuanlan.zhihu.com/p/4741261658, p16, equ-172, 174)
inline Matrix4d T_Exp(Vector6d xi){
    Vector3d pho = xi.head(3);
    Vector3d theta = xi.tail(3);
    Matrix3d V;
    double n = theta.norm();
    if (n < 1e-6){
        V = Matrix3d::Identity();
    } else {
        V = Matrix3d::Identity() + ( (1-cos(n))/(n*n) ) *hat(theta) + ( (n-sin(n))/(n*n*n) ) *hat(theta)*hat(theta);
    }
    Matrix4d T;
    T << R_Exp(theta), V*pho,
         0,0,0,                     1;
    return T;
}

// A micro Lie theory for state estimation in robotics (https://zhuanlan.zhihu.com/p/4741261658, p16, equ-173, 174, 145, 146)
inline Vector6d T_Log(Matrix4d T){
    T_isHomoTrans(T);
    Eigen::Matrix3d R = T.block(0,0,3,3);
    Eigen::Vector3d p = T.block(0,3,3,1);
    Matrix3d V_inv;
    Vector3d theta = R_Log(R);
    double n = theta.norm();
    if (n < 1e-6){
        V_inv = Matrix3d::Identity();
    } else {
        V_inv = Matrix3d::Identity() - 0.5*hat(theta) + ( 1/(n*n) - ( 1+cos(n) )/( 2*n*sin(n) ) ) * hat(theta)*hat(theta);
    } 
    Vector6d xi;
    xi <<   V_inv*p,
            theta;
    return xi;
}

// !!! Do not directly use this to do pd control because the translation part here are not exactly (p_des-p) 
// err = Log6(T'*T_des)
inline Vector6d T_boxminusL(Eigen::Matrix4d T_des, Eigen::Matrix4d T){
    T_isHomoTrans(T_des);
    T_isHomoTrans(T);
    return T_Log(T_inv(T)*T_des);
}

// err = Log6(T_des*T')
inline Vector6d T_boxminusW(Eigen::Matrix4d T_des, Eigen::Matrix4d T){
    T_isHomoTrans(T_des);
    T_isHomoTrans(T);
    return T_Log(T_des*T_inv(T));
}

}
