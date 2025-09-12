#include "legged_wbc/LeggedState.h"
#include "logger/CsvLogger.h"

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pinocchio/math/rpy.hpp>

// private
std::map<std::string, Eigen::VectorXd> LeggedState::getStateMap() const {
    std::map<std::string, Eigen::VectorXd> state_map;

    state_map["base_pos"] = base_pos_;
    state_map["base_R"] = Eigen::Map<const Eigen::VectorXd>(base_R_.data(), 9); // 3x3 → 9x1
    state_map["base_quat"] = base_quat_.coeffs();
    state_map["base_eulerZYX"] = base_eulerZYX_;
    state_map["base_lin_vel_W"] = base_lin_vel_W_;
    state_map["base_lin_vel_B"] = base_lin_vel_B_;
    state_map["base_ang_vel_W"] = base_ang_vel_W_;
    state_map["base_ang_vel_B"] = base_ang_vel_B_;
    state_map["base_eulerZYX_dot"] = base_eulerZYX_dot_;
    state_map["joint_pos"] = joint_pos_;
    state_map["joint_vel"] = joint_vel_;

    return state_map;
}

void LeggedState::updateCustomState() {
    // 先获取主状态 map
    std::map<std::string, Eigen::VectorXd> state_map = getStateMap();

    for (auto& custom_state_pair : custom_states_) {
        CustomState& custom_state = custom_state_pair.second;
        // 拼接
        int pos = 0;
        for (const auto& elem_name : custom_state.elements) {
            const Eigen::VectorXd& vec = state_map.at(elem_name);
            custom_state.state_vec.segment(pos, vec.size()) = vec;
            pos += vec.size();
        }
    }
}

// static
Eigen::Vector3d LeggedState::eulerZYX2AngularVelocityW(Eigen::Vector3d eulerZYX, Eigen::Vector3d eulerZYX_dot) {
  const double sz = sin(eulerZYX[0]);
  const double cz = cos(eulerZYX(0));
  const double sy = sin(eulerZYX(1));
  const double cy = cos(eulerZYX(1));
  const double dz = eulerZYX_dot(0);
  const double dy = eulerZYX_dot(1);
  const double dx = eulerZYX_dot(2);
  return { -sz * dy + cy * cz * dx, cz * dy + cy * sz * dx, dz - sy * dx };
}

Eigen::Vector3d LeggedState::AngularVelocityW2eulerZYX(Eigen::Vector3d eulerZYX, Eigen::Vector3d base_ang_vel_W) {
  const double sz = sin(eulerZYX(0));
  const double cz = cos(eulerZYX(0));
  const double sy = sin(eulerZYX(1));
  const double cy = cos(eulerZYX(1));
  const double wx = base_ang_vel_W(0);
  const double wy = base_ang_vel_W(1);
  const double wz = base_ang_vel_W(2);
  const double tmp = cz * wx / cy + sz * wy / cy;
  return {sy * tmp + wz, -sz * wx + cz * wy, tmp};
}

// public
// --- 构造函数实现 ---
// 构造函数：仅作为数据容器
void LeggedState::init(int num_joints) {
    // 默认初始化所有状态为零或单位矩阵/四元数
    base_pos_.setZero();
    base_R_.setIdentity();
    base_quat_.setIdentity();
    base_eulerZYX_.setZero();
    base_lin_vel_W_.setZero();
    base_ang_vel_W_.setZero();
    base_lin_vel_B_.setZero();
    base_ang_vel_B_.setZero();
    joint_pos_.setZero();
    joint_vel_.setZero();

    // 初始化关节向量大小
    num_joints_ = num_joints;
    joint_pos_.resize(num_joints_);
    joint_vel_.resize(num_joints_);

    rbd_state_.resize(getRbdStateSize());
}

void LeggedState::clear(){
    setBasePos(Eigen::Vector3d::Zero());
    setBaseRotationFromQuaternion(Eigen::Quaternion<double>::Identity());
    setBaseLinearVelocityW(Eigen::Vector3d::Zero());
    setBaseAngularVelocityW(Eigen::Vector3d::Zero());
    setJointPos(Eigen::VectorXd::Zero(num_joints_));
    setJointVel(Eigen::VectorXd::Zero(num_joints_));
}

void LeggedState::log(std::string prefix){
    CsvLogger& logger = CsvLogger::getInstance();

    // --- 主状态 ---
    logger.update(prefix+"base_pos", base_pos_);
    logger.update(prefix+"base_eulerZYX", static_cast<Eigen::VectorXd>(base_eulerZYX_));
    logger.update(prefix+"base_lin_vel_W", base_lin_vel_W_);
    logger.update(prefix+"base_lin_vel_B", base_lin_vel_B_);
    logger.update(prefix+"base_ang_vel_W", base_ang_vel_W_);
    logger.update(prefix+"base_ang_vel_B", base_ang_vel_B_);
    logger.update(prefix+"base_eulerZYX_dot", static_cast<Eigen::VectorXd>(base_eulerZYX_dot_));
    logger.update(prefix+"joint_pos", joint_pos_);
    logger.update(prefix+"joint_vel", joint_vel_);

    // base_R 展平为 9 维向量
    logger.update(prefix+"base_R", static_cast<Eigen::MatrixXd>(base_R_));

    // base_quat 作为 4 维向量
    Eigen::VectorXd quat_vec(4);
    quat_vec << base_quat_.x(), base_quat_.y(), base_quat_.z(), base_quat_.w();
    logger.update(prefix+"base_quat", quat_vec);

    logger.update(prefix+"rbd_state", rbd_state()); // rbd_state() 会自动更新

    // --- 自定义状态 ---
    for (auto& kv : custom_states_) {
        const std::string& name = kv.first;
        CustomState& cs = kv.second;       // 引用，避免拷贝
        updateCustomState();               // 确保 state_vec 最新
        logger.update(prefix+name, cs.state_vec); // 直接传 Eigen::VectorXd
    }
}

// --- 姿态更新方法实现 ---
void LeggedState::setBaseRotationFromMatrix(const Eigen::Matrix3d& R) {
    base_R_ = R;
    base_quat_ = Eigen::Quaterniond(R);
    base_eulerZYX_ = pinocchio::rpy::matrixToRpy(base_R_).reverse(); // ZYX 欧拉角
}

void LeggedState::setBaseRotationFromQuaternion(const Eigen::Quaterniond& quat) {
    base_quat_ = quat.normalized(); // 确保四元数是单位四元数
    base_R_ = base_quat_.toRotationMatrix();
    
    base_eulerZYX_ = pinocchio::rpy::matrixToRpy(base_R_).reverse(); // ZYX 欧拉角
}

void LeggedState::setBaseRotationFromQuaternion(const Eigen::VectorXd& quat) {
    if (quat.size() != 4) {
      throw std::runtime_error("[LeggedState]: Quaternion vector must be of size 4.");
    }

    base_quat_ = Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2)).normalized(); // 注意：Eigen::Quaterniond 的构造函数顺序是 (w, x, y, z)
    base_R_ = base_quat_.toRotationMatrix();
    base_eulerZYX_ = pinocchio::rpy::matrixToRpy(base_R_).reverse(); // ZYX 欧拉角
}

void LeggedState::setBaseRotationFromEulerZYX(const Eigen::Vector3d& eulerZYX) {
    base_eulerZYX_ = eulerZYX;
    Eigen::AngleAxisd yaw(base_eulerZYX_[0], Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitch(base_eulerZYX_[1], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd roll(base_eulerZYX_[2], Eigen::Vector3d::UnitX());

    base_quat_ = yaw * pitch * roll;
    base_R_ = base_quat_.toRotationMatrix();
}

// --- 线速度更新方法实现 ---
void LeggedState::setBaseLinearVelocityW(const Eigen::Vector3d& lin_vel_W) {
    base_lin_vel_W_ = lin_vel_W;
    base_lin_vel_B_ = base_R_.transpose() * base_lin_vel_W_;
}

void LeggedState::setBaseLinearVelocityB(const Eigen::Vector3d& lin_vel_B) {
    base_lin_vel_B_ = lin_vel_B;
    base_lin_vel_W_ = base_R_ * base_lin_vel_B_;
}

// --- 角速度更新方法实现 ---
void LeggedState::setBaseAngularVelocityW(const Eigen::Vector3d& ang_vel_W) {
    base_ang_vel_W_ = ang_vel_W;
    base_ang_vel_B_ = base_R_.transpose() * base_ang_vel_W_;
    base_eulerZYX_dot_ = AngularVelocityW2eulerZYX(base_eulerZYX_, base_ang_vel_W_);
}

void LeggedState::setBaseAngularVelocityB(const Eigen::Vector3d& ang_vel_B) {
    base_ang_vel_B_ = ang_vel_B;
    base_ang_vel_W_ = base_R_ * base_ang_vel_B_;
    base_eulerZYX_dot_ = AngularVelocityW2eulerZYX(base_eulerZYX_, base_ang_vel_W_);
}

void LeggedState::setBaseEulerZYXDot(const Eigen::Vector3d& base_eulerZYX_dot) {
    base_eulerZYX_dot_ = base_eulerZYX_dot;
    base_ang_vel_W_ = eulerZYX2AngularVelocityW(base_eulerZYX_, base_eulerZYX_dot_);
    base_ang_vel_B_ = base_R_.transpose() * base_ang_vel_W_;
}

// --- 完整更新方法实现 ---

void LeggedState::setFromRbdState(const Eigen::VectorXd& rbd_state) {
    if (rbd_state.size() != getRbdStateSize()) {
        std::cerr << "Error: setFromRbd received a state vector of incorrect size." << std::endl;
        return;
    }

    setBaseRotationFromEulerZYX(rbd_state.segment<3>(0));
    setBasePos(rbd_state.segment<3>(3));
    setJointPos(rbd_state.segment(6, num_joints_));
    setBaseAngularVelocityW(rbd_state.segment<3>(6 + num_joints_));
    setBaseLinearVelocityW(rbd_state.segment<3>(9 + num_joints_));
    setJointVel(rbd_state.segment(12 + num_joints_, num_joints_));
}

int LeggedState::getCustomeStateSize(const std::vector<std::string>& state_elements) {
    std::map<std::string, Eigen::VectorXd> state_map = getStateMap();

    int total_size = 0;
    for (const auto& elem_name : state_elements) {
        if (state_map.find(elem_name) == state_map.end()) {
            throw std::runtime_error("Element not found in state_map: " + elem_name);
        }
        total_size += state_map[elem_name].size();
    }
    return total_size;
}

void LeggedState::setFromCustomState(const std::string& state_name, const Eigen::VectorXd& state_vec) {
    // 检查 custom state 是否存在
    auto it = custom_states_.find(state_name);
    if (it == custom_states_.end()) {
        throw std::runtime_error("Custom state not found: " + state_name);
    }

    CustomState& custom_state = it->second;

    // 确认长度匹配
    if (state_vec.size() != getCustomeStateSize(custom_state.elements)) {
        throw std::runtime_error("state_vec size mismatch for custom state: " + state_name);
    }

    // 逐段更新主状态
    std::map<std::string, Eigen::VectorXd> state_map = getStateMap();
    int pos = 0;
    for (const auto& elem_name : custom_state.elements) {
        Eigen::VectorXd segment;
        int seg_size = state_map[elem_name].size();
        segment = state_vec.segment(pos, seg_size);
        pos += seg_size;

        // 更新对应主状态
        if (elem_name == "base_pos") {
            setBasePos(segment);
        } else if (elem_name == "base_R") {
            if (seg_size != 9) throw std::runtime_error("base_R segment size error");
            setBaseRotationFromMatrix(Eigen::Map<const Eigen::Matrix3d>(segment.data()));
        } else if (elem_name == "base_quat") {
            if (seg_size != 4) throw std::runtime_error("base_quat segment size error");
            setBaseRotationFromQuaternion(segment);
        } else if (elem_name == "base_eulerZYX") {
            setBaseRotationFromEulerZYX(segment);
        } else if (elem_name == "base_lin_vel_W") {
            setBaseLinearVelocityW(segment);
        } else if (elem_name == "base_lin_vel_B") {
            setBaseLinearVelocityB(segment);
        } else if (elem_name == "base_ang_vel_W") {
            setBaseAngularVelocityW(segment);
        } else if (elem_name == "base_ang_vel_B") {
            setBaseAngularVelocityB(segment);
        } else if (elem_name == "base_eulerZYX_dot") {
            setBaseEulerZYXDot(segment);
        } else if (elem_name == "joint_pos") {
            setJointPos(segment);
        } else if (elem_name == "joint_vel") {
            setJointVel(segment);
        } else {
            throw std::runtime_error("Unknown element: " + elem_name);
        }
    }
}
