#include "legged_wbc/Lie.h"

#include "iostream"
#include <Eigen/src/Geometry/Quaternion.h>
#include <cstdio>
#include <manif/impl/se3/SE3.h>
#include <manif/impl/so3/SO3.h>
#include <manif/manif.h>

using namespace Lie;

int main(){
    std::getchar();
    std::srand((unsigned int) std::time(nullptr));

    Eigen::Quaterniond q1 = Eigen::Quaterniond::UnitRandom();
    Eigen::Quaterniond q2 = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector4d q1_, q2_;
    q1_ << q1.w(), q1.x(), q1.y(), q1.z();
    q2_ << q2.w(), q2.x(), q2.y(), q2.z();

    std::cout << "[Lie_test] q1: " << q1.coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1_: " << q1_.transpose() << std::endl;
    std::cout << "[Lie_test] q2: " << q2.coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q2_: " << q2_.transpose() << std::endl;

    std::cout << "[Lie_test] q1*q2 (Eigen):    " << (q1*q2).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1*q2 (quat_productMatL): " << quat_product(q1_, q2_).transpose() << std::endl;
    std::cout << "[Lie_test] q1*q2 (quat_productMatR): " << ( quat_productMatR(q2_) * q1_).transpose() << std::endl;


    std::cout << "[Lie_test] q1  conjugate: " << q1.conjugate().coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1_ conjugate: " << quat_conjugate(q1_).transpose() << std::endl;

    Eigen::Vector3d phi1 = quat_Log(q1_);
    std::cout << "[Lie_test] Log(q1_): " << phi1.transpose() << std::endl;
    std::cout << "[Lie_test] Exp(Log(q1_)): (Eigen) " << Eigen::Quaterniond(Eigen::AngleAxisd(phi1.norm(), phi1.normalized())).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Exp(Log(q1_)): (quat_Exp) " << quat_Exp(quat_Log(q1_)).transpose() << std::endl;

    Eigen::Matrix3d R1 = q1.toRotationMatrix();
    Eigen::Matrix3d R2 = q2.toRotationMatrix();
    Eigen::Matrix3d R1_ = quat_ToR(q1_);
    Eigen::Matrix3d R2_ = quat_ToR(q2_);
    std::cout << "[Lie_test] R1:\n" << R1 << std::endl;
    std::cout << "[Lie_test] R1_:\n" << R1_ << std::endl;

    Eigen::Vector3d phi1_R = R_Log(R1_);
    std::cout << "[Lie_test] Exp(Log(R1_)): (Eigen)\n" << Eigen::Quaterniond(Eigen::AngleAxisd(phi1_R.norm(), phi1_R.normalized())).toRotationMatrix() << std::endl;
    std::cout << "[Lie_test] Exp(Log(R1_)): (R_Exp)\n" << R_Exp(R_Log(R1_)) << std::endl;
    
    Eigen::Quaterniond q1_R(R1);
    std::cout << "[Lie_test] q1_R: (Eigen)" << q1_R.coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] q1_R: (R_ToQuat)" << R_ToQuat(R1_).transpose() << std::endl;

    std::cout << "[Lie_test] q1_*q2':\n" << quat_ToR(quat_product(q1_, quat_conjugate(q2_))) << std::endl;
    std::cout << "[Lie_test] R1_ * R2_^T:\n" << R1_*R2_.transpose() << std::endl;

    manif::SO3d SO3_1(q1), SO3_2(q2);
    std::cout << "[Lie_test] manif\n";
    std::cout << "[Lie_test] Log(R1*R2^T): " << SO3_1.lminus(SO3_2).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2*R1^T): " << SO3_2.lminus(SO3_1).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2^T*R1): " << SO3_1.rminus(SO3_2).coeffs().transpose() << std::endl;
    std::cout << "[Lie_test] Log(R1^T*R_2): " << SO3_2.rminus(SO3_1).coeffs().transpose() << std::endl;
    
    std::cout << "[Lie_test] self\n";
    std::cout << "[Lie_test] Log(R1*R2^T): " << R_boxminusW(R1_, R2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2*R1^T): " << R_boxminusW(R2_, R1_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(R2^T*R1): " << R_boxminusL(R1_, R2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(R1^T*R_2): " << R_boxminusL(R2_, R1_).transpose() << std::endl;
    std::cout << "[Lie_test] R2*Log(R2^T*R1): " << (R2_*R_boxminusL(R1_, R2_)).transpose() << std::endl;
    std::cout << "[Lie_test] R1*Log(R1^T*R_2): " << (R1_*R_boxminusL(R2_, R1_)).transpose() << std::endl;

    std::cout << "[Lie_test] Log(q1*q2'): " << quat_boxminusW(q1_,q2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(q2*q1'): " << quat_boxminusW(q2_,q1_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(q2'*q1): " << quat_boxminusL(q1_,q2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(q1'*q2): " << quat_boxminusL(q2_,q1_).transpose() << std::endl;
    std::cout << "[Lie_test] q2*Log(q2'*q1)*q2': " << ( quat_rotateVec(q2_, quat_boxminusL(q1_,q2_).transpose()) ).transpose() << std::endl;
    std::cout << "[Lie_test] q1*Log(q1'*q2)*q1': " << ( quat_rotateVec(q1_, quat_boxminusL(q2_,q1_).transpose()) ).transpose() << std::endl;

    manif::SE3d T1 = manif::SE3d::Random();
    manif::SE3d T2 = manif::SE3d::Random();
    T1.quat(q1);
    T2.quat(q2);
    Matrix4d T1_ = T1.transform();
    Matrix4d T2_ = T2.transform();
    std::cout << "[Lie_test] T1:\n" << T1 << std::endl;
    std::cout << "[Lie_test] T2:\n" << T2 << std::endl;
    std::cout << "[Lie_test] T1_:\n" << T1_ << std::endl;
    std::cout << "[Lie_test] T2_:\n" << T2_ << std::endl;

    std::cout << "[Lie_test] Log(T1):" << T1.log() << std::endl;
    std::cout << "[Lie_test] Log(T1_):" << T_Log(T1_).transpose() << std::endl;
    std::cout << "[Lie_test] Exp(Log(T1)):\n" << T1.log().exp().transform() << std::endl;
    std::cout << "[Lie_test] Exp(Log(T1_)):\n" << T_Exp(T_Log(T1_)) << std::endl;
    
    std::cout << "[Lie_test] Log(T2'*T1):" << T1.rminus(T2) << std::endl;
    std::cout << "[Lie_test] Log(T1'*T2):" << T2.rminus(T1) << std::endl;
    std::cout << "[Lie_test] Log(T1*T2'):" << T1.lminus(T2) << std::endl;
    std::cout << "[Lie_test] Log(T2*T1'):" << T2.lminus(T1) << std::endl;

    std::cout << "[Lie_test] Log(T2'*T1):" << T_boxminusL(T1_, T2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(T1'*T2):" << T_boxminusL(T2_, T1_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(T1*T2'):" << T_boxminusW(T1_, T2_).transpose() << std::endl;
    std::cout << "[Lie_test] Log(T2*T1'):" << T_boxminusW(T2_, T1_).transpose() << std::endl;
}
