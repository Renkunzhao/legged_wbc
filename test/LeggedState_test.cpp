#include "legged_wbc/LeggedState.h"

#include <iostream>
#include <Eigen/Dense>

#include <pinocchio/math/rpy.hpp>

#include "Eigen/src/Geometry/Quaternion.h"

const int NUM_JOINTS = 12;

int main() {
    // debug: 方便 attach
    std::getchar();

    Eigen::AngleAxisd yaw(0.1, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitch(0.1, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd roll(0.1, Eigen::Vector3d::UnitX());
  
    std::cout << "--- Eigen::AngleAxisd ---" << std::endl;
    std::cout << "yaw\n" << yaw.matrix() 
              << "\npitch\n" << pitch.matrix() 
              << "\nroll\n" << roll.matrix() 
              << std::endl;

    LeggedState state_container(NUM_JOINTS);
    // 姿态
    // 随机给定 eulerZYX, 通过 setBaseRotationFromEulerZYX 设置为基座姿态
    // 计算旋转矩阵 base_R，再通过 setBaseRotationFromMatrix 设置为姿态
    // 比较计算出的 eulerZYX 和 原来的是否相同，验证 setBaseRotationFromEulerZYX 和 setBaseRotationFromMatrix 是否互逆
    Eigen::Vector3d eulerZyx = Eigen::Vector3d::Random() * (M_PI - 0.05);
    std::cout << "euler (origin):" << eulerZyx.transpose() << std::endl;
    state_container.setBaseRotationFromEulerZYX(eulerZyx);
    state_container.setBaseRotationFromMatrix(state_container.base_R());
    std::cout << "euler (Matrix):" << state_container.base_eulerZYX().transpose() << std::endl;

    // 同样的方式，验证 setBaseRotationFromQuaternion 和 setBaseRotationFromMatrix setBaseRotationFromEulerZYX 是否互逆
    Eigen::Quaterniond quat = Eigen::Quaterniond::UnitRandom();
    std::cout << "quat (origin):" << quat.coeffs().transpose() << std::endl;
    state_container.setBaseRotationFromQuaternion(quat);
    state_container.setBaseRotationFromMatrix(state_container.base_R());
    std::cout << "quat (Matrix):" << state_container.base_quat().coeffs().transpose() << std::endl;
    state_container.setBaseRotationFromQuaternion(quat);
    state_container.setBaseRotationFromEulerZYX(state_container.base_eulerZYX());
    std::cout << "quat (euler):" << state_container.base_quat().coeffs().transpose() << std::endl;

    // 比较 FBRS 和 pinocchio::rpy 计算出的旋转矩阵是否相同
    // 注意： pinocchio::rpy 中也是 ZYX-euler , 但存储顺序为 (roll, pitch, yaW)
    state_container.setBaseRotationFromEulerZYX(eulerZyx);
    std::cout << "R (FBRS): \n" << state_container.base_R() << std::endl;
    std::cout << "R (Pino): \n" << pinocchio::rpy::rpyToMatrix(eulerZyx.reverse()) << std::endl;

    // 角速度
    // 随机给定 eulerZyx_dot, 比较 FBRS 和 pinocchio::rpy 计算出的世界、身体坐标系下的角速度是否相同
    Eigen::Vector3d eulerZyx_dot = Eigen::Vector3d::Random() * 10;
    state_container.setBaseEulerZYXDot(eulerZyx_dot);
    auto rpy_J_W = pinocchio::rpy::computeRpyJacobian(state_container.base_eulerZYX().reverse(), pinocchio::LOCAL_WORLD_ALIGNED);
    auto rpy_J_B = pinocchio::rpy::computeRpyJacobian(state_container.base_eulerZYX().reverse(), pinocchio::LOCAL);
    std::cout << "ang_vel_w (FBRS):" << state_container.base_ang_vel_W().transpose() << std::endl;
    std::cout << "ang_vel_w (pino):" << (rpy_J_W * eulerZyx_dot.reverse()).transpose() << std::endl;
    std::cout << "ang_vel_B (FBRS):" << state_container.base_ang_vel_B().transpose() << std::endl;
    std::cout << "ang_vel_B (pino):" << (rpy_J_B * eulerZyx_dot.reverse()).transpose() << std::endl;

    // Custom State
    // 创建一个 custom_state = rbd_state = [base_eulerZYX, base_pos, joint_pos, base_ang_vel_W_, base_lin_vel_W_, joint_vel]
    // 通过比较二者是否相等， 测试 setFromCustomState ，custom_state 等函数功能
    state_container.createCustomState("rbd_state", 
      {"base_eulerZYX", "base_pos", "joint_pos", "base_ang_vel_W", "base_lin_vel_W", "joint_vel"});
    
    // 位置
    // 测试 setFromRbdState
    Eigen::VectorXd state = Eigen::VectorXd::Random(state_container.getRbdStateSize());
    std::cout << "rbdstate (origin):   " << state.transpose() << std::endl;

    state_container.setFromRbdState(state);
    auto rbd_state = state_container.rbd_state();
    std::cout << "rbdstate (rbd):  " << rbd_state.transpose() << std::endl;
    std::cout << "error:" << (state - rbd_state).norm() << std::endl;

    // 测试 setFromCustomState
    state_container.setFromCustomState("rbd_state",state);
    auto custom_state = state_container.custom_state("rbd_state");
    std::cout << "rbdstate (custom):  " << custom_state.transpose() << std::endl;
    std::cout << "error:" << (state - custom_state).norm() << std::endl;

    return 0;
}
