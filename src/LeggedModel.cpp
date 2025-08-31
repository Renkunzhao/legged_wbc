#include "legged_wbc/LeggedModel.h"
#include "pinocchio/multibody/fwd.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

void LeggedModel::loadUrdf(std::string urdfPath, std::string baseType, 
                            std::string baseName, std::vector<std::string> contact3DofNames, std::vector<std::string> contact6DofNames){
    std::cout << "[LeggedModel] Load URDF from " << urdfPath << std::endl;
    if (baseType == "quaternion") {
        // 使用 pinocchio::JointModelFreeFlyer 的浮动基机器人模型（基于四元数）
        pinocchio::urdf::buildModel(urdfPath, pinocchio::JointModelFreeFlyer(), model_);
    } else if (baseType == "eulerZYX") {
        // 使用 pinocchio::JointModelComposite(Translation + EulerZYX) 的浮动基机器人模型   
        pinocchio::JointModelComposite jointComposite(2);
        jointComposite.addJoint(pinocchio::JointModelTranslation());      // 3 DoF 平移
        jointComposite.addJoint(pinocchio::JointModelSphericalZYX());     // 3 DoF 旋转
        pinocchio::urdf::buildModel(urdfPath, jointComposite, model_);
    } else {
        throw std::runtime_error("Invalid orientation type specified: " + baseType);
    }

    data_ = pinocchio::Data(model_);    

    nJoints_ = model_.nv - 6;
    leggedState_.init(nJoints_);

    baseName_ = baseName;

    contact3DofNames_ = contact3DofNames;
    nContacts3Dof_ = contact3DofNames_.size();
    for(const auto& ee3Dof_ : contact3DofNames_) contact3DofIds_.push_back(model_.getBodyId(ee3Dof_));

    contact6DofNames_ = contact6DofNames;
    nContacts6Dof_ = contact6DofNames_.size();
    for(const auto& ee6Dof_ : contact6DofNames_) contact6DofIds_.push_back(model_.getBodyId(ee6Dof_));

    // init();  // 初始化 q 和 v    
    // setLimits();  // 设置关节位置和速度的上下限
}

std::vector<Eigen::Vector3d> LeggedModel::contact3DofPoss(const Eigen::VectorXd& q_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    std::vector<Eigen::Vector3d> contact3DofPoss;
    for (const auto& Id : contact3DofIds_) contact3DofPoss.push_back(data_.oMf[Id].translation());
    return contact3DofPoss;
}

std::vector<Eigen::Vector3d> LeggedModel::contact3DofVels(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);

    std::vector<Eigen::Vector3d> contact3DofVels;
    for (const auto& Id : contact3DofIds_) contact3DofVels.push_back(pinocchio::getFrameVelocity(model_, data_, Id, pinocchio::LOCAL_WORLD_ALIGNED).linear());
    return contact3DofVels;
}

std::vector<Eigen::Vector3d> LeggedModel::contact6DofPoss(const Eigen::VectorXd& q_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    std::vector<Eigen::Vector3d> contact6DofPoss;
    for (const auto& Id : contact6DofIds_) contact6DofPoss.push_back(data_.oMf[Id].translation());
    return contact6DofPoss;
}

std::vector<Eigen::Vector3d> LeggedModel::contact6DofVels(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);

    std::vector<Eigen::Vector3d> contact6DofVels;
    for (const auto& Id : contact6DofIds_) contact6DofVels.push_back(pinocchio::getFrameVelocity(model_, data_, Id, pinocchio::LOCAL_WORLD_ALIGNED).linear());
    return contact6DofVels;
}
