#ifndef LEGGEDSTATE_H
#define LEGGEDSTATE_H

#include <string>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Geometry>

/**
 * @class LeggedState
 * @brief 浮动基机器人状态类，封装了位置、姿态、速度和动量等所有状态数据，并提供自动更新的逻辑。
 *        使用 Eigen 库进行矩阵、向量和四元数运算，确保数据同步。
 *
 * @warning：该类目前仍在测试中，以下是已测试完成的函数列表：
 * - [ ]
 * - [ ]
 */

class LeggedState {
private:
    // --- 机器人状态数据成员 (私有) ---
    
    // 浮动基位置
    Eigen::Vector3d base_pos_;

    // 浮动基姿态（三种表示形式 ，始终保持同步)
    Eigen::Matrix3d base_R_;
    Eigen::Quaterniond base_quat_;
    Eigen::Vector3d base_eulerZYX_; // (yaw, pitch, roll) ZYX 欧拉角

    // 浮动基线速度
    Eigen::Vector3d base_lin_vel_W_;
    Eigen::Vector3d base_lin_vel_B_;

    // 浮动基角速度
    Eigen::Vector3d base_ang_vel_W_;
    Eigen::Vector3d base_ang_vel_B_;
    Eigen::Vector3d base_eulerZYX_dot_;
    
    // 关节状态
    size_t num_joints_;
    std::vector<std::string> joint_names_;
    Eigen::VectorXd joint_pos_;
    Eigen::VectorXd joint_vel_;

    Eigen::VectorXd rbd_state_;

    // 内部调用， 自动更新rbd_state
    void updateRbdState() {rbd_state_ << base_eulerZYX_, base_pos_, joint_pos_, base_ang_vel_W_, base_lin_vel_W_, joint_vel_;}

    struct CustomState {
        std::vector<std::string> elements;
        Eigen::VectorXd state_vec;
        std::vector<std::string> joint_order;
    };

    std::map<std::string, CustomState> custom_states_;

    // 内部调用， 生成 state_map 便于操作 CustomState
    std::map<std::string, Eigen::VectorXd> getStateMap() const ;

    /**
     * @brief 内部调用， 自动更新 custom_state。
     * @param 示例 如： custom_state.second.elements = {"base_pos", "base_quat", "joint_pos"}
     *            则: custom_state.second.state_vec << base_pos_, base_quat_, joint_pos
     */
    void updateCustomState();

public:
    /**
    * Compute angular velocities expressed in the world frame from derivatives of ZYX-Euler angles
    *
    * @param [in] eulerZYX: ZYX-Euler angles
    * @param [in] eulerZYX_dot: time-derivative of ZYX-Euler angles
    * @return angular velocity expressed in world frame
    */
    static Eigen::Vector3d eulerZYX2AngularVelocityW(Eigen::Vector3d eulerZYX, Eigen::Vector3d eulerZYX_dot);

    /**
    * Compute derivatives of ZYX-Euler angles from global angular velocities
    * The inverse of getGlobalAngularVelocityFromEulerAnglesZyxDerivatives
    *
    * This transformation is singular for y = +- pi / 2
    *
    * @param [in] eulerZYX: ZYX-Euler angles
    * @param [in] base_ang_vel_W: angular velocity expressed in world frame
    * @return derivatives of ZYX-Euler angles
    */
    static Eigen::Vector3d AngularVelocityW2eulerZYX(Eigen::Vector3d eulerZYX, Eigen::Vector3d base_ang_vel_W);

    /**
     * @brief 构造函数，用于仅作为数据容器。
     * @param num_joints 机器人关节数
     */
    LeggedState() = default;
    LeggedState(int num_joints, std::vector<std::string> joint_names) {init(num_joints, joint_names);}
    void init(int num_joints, std::vector<std::string> joint_names);

    /**
     * @brief 创建自定义状态。
     * @param state_name 机器人关节数
     * @param state_elements 该状态包含的元素名称列表,可选的有：
     *          {"base_pos",
     *           "base_R",
     *           "base_quat",
     *           "base_eulerZYX",
     *           "base_lin_vel_W",
     *           "base_ang_vel_W",
     *           "base_lin_vel_B",
     *           "base_ang_vel_B",
     *           "base_eulerZYX_dot",
     *           "joint_pos",
     *           "joint_vel"}
     */
    void createCustomState(const std::string& state_name, const std::vector<std::string>& state_elements, std::vector<std::string> joint_order = {}) {
        if (custom_states_.count(state_name)) {
            throw std::runtime_error("Custom state already exists: " + state_name);
        }

        CustomState state;
        state.elements = state_elements;
        state.state_vec.resize(getCustomeStateSize(state_elements));
        state.joint_order = joint_order;    
        custom_states_[state_name] = std::move(state);
    }

    /**
     * @brief 重置所有状态为默认值
     */
    void clear();

    /**
     * @brief 根据当前时间记录所有状态
     */
    void log(std::string prefix);

    /**
     * @brief 设置浮动基位置。
     * @param base_pos 浮动基在世界坐标系下的位置
     */
    void setBasePos(const Eigen::Vector3d& base_pos) { base_pos_ = base_pos; }

    // --- 姿态更新方法 (设置器/setter) ---
    /**
     * @brief 根据旋转矩阵更新姿态，并自动同步四元数和欧拉角。
     * @param R 浮动基在世界坐标系下的旋转矩阵
     */
    void setBaseRotationFromMatrix(const Eigen::Matrix3d& R);
    
    /**
     * @brief 根据四元数更新姿态，并自动同步旋转矩阵和欧拉角。
     * @param quat 浮动基在世界坐标系下的四元数
     */
    void setBaseRotationFromQuaternion(const Eigen::Quaterniond& quat);

    /**
     * @brief 根据四元数更新姿态，并自动同步旋转矩阵和欧拉角。
     * @param quat 浮动基在世界坐标系下的四元数 [x, y, z, w]
     */
    void setBaseRotationFromQuaternion(const Eigen::VectorXd& quat);

    /**
     * @brief 根据欧拉角更新姿态，并自动同步旋转矩阵和四元数。
     * @param eulerZYX 浮动基在世界坐标系下的欧拉角 (ZYX)
     */
    void setBaseRotationFromEulerZYX(const Eigen::Vector3d& eulerZYX);
    
    // --- 线速度更新方法 (设置器/setter) ---
    /**
     * @brief 根据世界坐标系下的线速度更新，并自动同步身体坐标系下的线速度。
     * @param lin_vel_W 世界坐标系下的线速度
     */
    void setBaseLinearVelocityW(const Eigen::Vector3d& lin_vel_W);

    /**
     * @brief 根据身体坐标系下的线速度更新，并自动同步世界坐标系下的线速度。
     * @param lin_vel_B 身体坐标系下的线速度
     */
    void setBaseLinearVelocityB(const Eigen::Vector3d& lin_vel_B);

    // --- 角速度更新方法 (设置器/setter) ---
    /**
     * @brief 根据世界坐标系下的角速度更新，并自动同步身体坐标系下的角速度。
     * @param ang_vel_W 世界坐标系下的角速度
     */
    void setBaseAngularVelocityW(const Eigen::Vector3d& ang_vel_W);

    /**
     * @brief 根据身体坐标系下的角速度更新，并自动同步世界坐标系下的角速度。
     * @param ang_vel_B 身体坐标系下的角速度
     */
    void setBaseAngularVelocityB(const Eigen::Vector3d& ang_vel_B);

    /**
     * @brief 根据ZYX euler 变化率更新，并自动同步世界、身体坐标系下的角速度。
     * @param base_eulerZYX_dot ZYX euler 变化率
     */
    void setBaseEulerZYXDot(const Eigen::Vector3d& base_eulerZYX_dot);

    /**
     * @brief 设置关节位置。
     * @param joint_pos 关节位置向量
     */
    void setJointPos(const Eigen::VectorXd& joint_pos, const std::vector<std::string>& joint_order = {});

    /**
     * @brief 设置关节速度。
     * @param joint_vel 关节速度向量
     */
    void setJointVel(const Eigen::VectorXd& joint_vel, const std::vector<std::string>& joint_order = {});
    
    // 完整状态更新
    /**
     * @brief 根据RBD状态向量更新所有主状态变量。
     * @param rbd_state [base_eulerZYX, base_pos, joint_pos, base_ang_vel_W_, base_lin_vel_W_, joint_vel]
     */
    void setFromRbdState(const Eigen::VectorXd& rbd_state);

    /**
     * @brief 根据自定义状态向量更新所有主状态变量。
     * @param state_name 
     * @param custom_state 
     */
    void setFromCustomState(const std::string& state_name, const Eigen::VectorXd& state_vec);

    // --- 数据访问方法 (获取器/getter) ---
    // 返回 const 引用，以避免数据不必要的拷贝，并防止外部修改
    const Eigen::Vector3d& base_pos() const { return base_pos_; }
    const Eigen::Matrix3d& base_R() const { return base_R_; }
    const Eigen::Quaterniond& base_quat() const { return base_quat_; }
    const Eigen::Vector3d& base_eulerZYX() const { return base_eulerZYX_; }
    const Eigen::Vector3d& base_lin_vel_W() const { return base_lin_vel_W_; }
    const Eigen::Vector3d& base_lin_vel_B() const { return base_lin_vel_B_; }
    const Eigen::Vector3d& base_ang_vel_W() const { return base_ang_vel_W_; }
    const Eigen::Vector3d& base_ang_vel_B() const { return base_ang_vel_B_; }
    const Eigen::Vector3d& base_eulerZYX_dot() const { return base_eulerZYX_dot_; }
    const Eigen::VectorXd& joint_pos() const { return joint_pos_; }
    const Eigen::VectorXd& joint_vel() const { return joint_vel_; }

    const std::vector<std::string>& joint_names() const {return joint_names_;}

    int getRbdStateSize() const { return 2*(num_joints_+6); }
    const Eigen::VectorXd& rbd_state() { updateRbdState(); return rbd_state_; }

    int getCustomeStateSize(const std::vector<std::string>& state_elements);
    const std::vector<std::string>& getCustomeJointOrder(const std::string& state_name) {
      return custom_states_.at(state_name).joint_order;
    }
    const Eigen::VectorXd& custom_state(const std::string& state_name) { 
      updateCustomState(); 
      return custom_states_.at(state_name).state_vec; 
    }

    static void reorderJoints(const Eigen::VectorXd& input,
                    const std::vector<std::string>& input_order,
                    const std::vector<std::string>& target_order,
                    Eigen::VectorXd& output) 
    {
        if (input.size() != (int)input_order.size() ||
            input_order.size() != target_order.size()) {
            throw std::runtime_error("[LeggedState] joint vector or order size mismatch.");
        }

        std::map<std::string, int> order_map;
        for (size_t i = 0; i < input_order.size(); ++i) {
            order_map[input_order[i]] = i;
        }

        output.resize(target_order.size());
        for (size_t i = 0; i < target_order.size(); ++i) {
            auto it = order_map.find(target_order[i]);
            if (it == order_map.end()) {
                throw std::runtime_error("[LeggedState] joint name " + target_order[i] + " not found in input_order.");
            }
            output[i] = input[it->second];
        }
    }
};

#endif // LEGGEDSTATE_H
