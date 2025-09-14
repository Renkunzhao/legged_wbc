#include "legged_wbc/LeggedModel.h"
#include "legged_wbc/Lie.h"
#include <cstddef>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/math/rpy.hpp>

using namespace Lie;

void LeggedModel::loadUrdf(std::string urdfPath, std::string baseType, std::string baseName,
                           std::vector<std::string> contact3DofNames, 
                           std::vector<std::string> contact6DofNames, 
                           bool verbose) {
    std::cout << "[LeggedModel] Load URDF from " << urdfPath << std::endl;
    baseType_ = baseType;
    if (baseType_ == "quaternion") {
        // 使用 pinocchio::JointModelFreeFlyer 的浮动基机器人模型（基于四元数）
        pinocchio::urdf::buildModel(urdfPath, pinocchio::JointModelFreeFlyer(), model_);
        nqBase_ = 7;
    } else if (baseType_ == "eulerZYX") {
        // 使用 pinocchio::JointModelComposite(Translation + EulerZYX) 的浮动基机器人模型   
        pinocchio::JointModelComposite jointComposite(2);
        jointComposite.addJoint(pinocchio::JointModelTranslation());      // 3 DoF 平移
        jointComposite.addJoint(pinocchio::JointModelSphericalZYX());     // 3 DoF 旋转
        pinocchio::urdf::buildModel(urdfPath, jointComposite, model_);
        nqBase_ = 6;
    } else {
        throw std::runtime_error("Invalid orientation type specified: " + baseType_);
    }

    std::cout << "---- Joints ----" << std::endl;
    for (size_t i = 0; i < model_.joints.size(); ++i) {
        std::cout << i << ": " << model_.names[i] << std::endl;
    }

    std::cout << "---- Links (Frames of type BODY) ----" << std::endl;
    for (size_t i = 0; i < model_.frames.size(); ++i) {
        if (model_.frames[i].type == pinocchio::BODY) {
            std::cout << i << ": " << model_.frames[i].name << std::endl;
        }
    }

    data_ = pinocchio::Data(model_);    

    nJoints_ = model_.nv - 6;

    // set joint order
    // 跳过 universe 和 base
    for (size_t i = 2; i < model_.names.size(); ++i) {
        jointNames_.push_back(model_.names[i]);
    }

    baseName_ = baseName;

    contact3DofNames_ = contact3DofNames;
    nContacts3Dof_ = contact3DofNames_.size();
    for(const auto& ee3Dof_ : contact3DofNames_) contact3DofIds_.push_back(model_.getBodyId(ee3Dof_));

    contact6DofNames_ = contact6DofNames;
    nContacts6Dof_ = contact6DofNames_.size();
    for(const auto& ee6Dof_ : contact6DofNames_) contact6DofIds_.push_back(model_.getBodyId(ee6Dof_));

    // Translation bounds
    model_.lowerPositionLimit.head<3>().setConstant(-10.0);  // x, y, z
    model_.upperPositionLimit.head<3>().setConstant(10.0);

    // Orientation（四元数不设置限制，EulerZYX可以设置为 -pi 到 pi）
    if (baseType_ == "eulerZYX") {
        model_.lowerPositionLimit.segment<3>(3).setConstant(-M_PI);
        model_.upperPositionLimit.segment<3>(3).setConstant(M_PI);
    }

    // 设置关节限制 避开奇异点
    for (int i=0;i<contact3DofNames.size();i++) {
        model_.lowerPositionLimit[nqBase_ + 3*i + 1] = -0.2094;
        model_.upperPositionLimit[nqBase_ + 3*i + 1] = 1.9897;
    }

    verbose_ = verbose;
    if (true) {
        std::cout << "[LeggedModel] nDof: " << nDof() << std::endl; 
    }
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

Eigen::VectorXd LeggedModel::inverseKine3Dof(Eigen::VectorXd qBase, const std::vector<Eigen::Vector3d>& contact3DofPoss){
    if (qBase.size() != nqBase_) {
        throw std::runtime_error("Base pose vector size does not match nqBase_");
    }
    if (contact3DofPoss.size() != contact3DofNames_.size()) {
        throw std::runtime_error("Mismatch in number of target positions and foot names");
    }

    // TODO don't use 
    Eigen::Matrix3d R;
    if (baseType_ == "quaternion") {
        // qBase xyzw, quat_ToR require wxyz
        R = quat_ToR(quat_wxyz(qBase.tail(4)));
    }
    else if(baseType_ == "eulerZYX") {
        R = pinocchio::rpy::rpyToMatrix(qBase.tail(3).reverse());
    }

    // contact3DofPoss is feet position in world frame, get foot pos relative to base in base frame using R^t * (contact3DofPoss - base_pos)
    Eigen::VectorXd desEEpos(nContacts3Dof_ * 3);
    for (size_t i = 0; i < contact3DofPoss.size(); i++) {
        desEEpos.segment(3*i, 3) = R.transpose() * (contact3DofPoss[i] - qBase.head(3));
    }

    int max_iters = 1000;
    double tol = 1e-4, dt = 0.1, damping = 1e-6;
    Eigen::VectorXd q = (model_.upperPositionLimit + model_.lowerPositionLimit)/2;
    if (verbose_) std::cout << "[LeggedModel] IK start from " << q.transpose() << std::endl;
    // err = [err_foot_1^T, err_foot_2^T, ...]^T
    Eigen::VectorXd err = Eigen::VectorXd::Zero(nContacts3Dof_*3);
    Eigen::VectorXd dqj = Eigen::VectorXd::Zero(model_.nv-6);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nContacts3Dof_*3, model_.nv-6);
    Eigen::MatrixXd Jt = Eigen::MatrixXd::Zero(model_.nv-6, nContacts3Dof_*3);
    Eigen::MatrixXd JJt = Eigen::MatrixXd::Zero(nContacts3Dof_*3, nContacts3Dof_*3);
    Eigen::MatrixXd JJt_damped = Eigen::MatrixXd::Zero(nContacts3Dof_*3, nContacts3Dof_*3);
    for (int i = 0; i < max_iters; i++) {
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);

        err = desEEpos;
        for (size_t i = 0; i < contact3DofIds_.size(); i++) {
            err.segment(3*i, 3) -= data_.oMf[contact3DofIds_[i]].translation();
        }

        if (err.norm() < tol) {
            if (verbose_) std::cout << "[LeggedModel] IK Converged in " << i << " iterations. Final error: " << err.norm() << std::endl;
            q.head(nqBase_) = qBase;
            return q;
        }
        
        for (size_t i = 0; i < contact3DofIds_.size(); i++) {
            Eigen::MatrixXd Jac(6, model_.nv);
            Jac.setZero();
            pinocchio::computeFrameJacobian(model_, data_, q, contact3DofIds_[i], pinocchio::LOCAL_WORLD_ALIGNED, Jac);
            J.block(3*i, 0, 3, J.cols()) = Jac.block(0, 6, 3, model_.nv-6);
        }

        Jt = J.transpose();
        JJt = J * Jt;
        JJt_damped = JJt + damping * Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols());
        dqj = Jt * (JJt_damped.ldlt().solve(err));
        
        q.tail(dqj.size()) += dqj * dt;

        // 将角度包裹到 [-pi, pi]
        for (int j = 0; j < dqj.size(); ++j) {
            double& angle = q[nqBase_ + j];
            angle = std::atan2(std::sin(angle), std::cos(angle));
        }
    }
}
