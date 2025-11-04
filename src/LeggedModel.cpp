#include "legged_wbc/LeggedModel.h"
#include "legged_wbc/Math.h"
#include "legged_wbc/Lie.h"
#include "legged_wbc/Yaml.h"
#include <cstddef>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/math/rpy.hpp>

using namespace Lie;

void LeggedModel::loadConfig(const YAML::Node& node){
    this->loadUrdf(node["urdfPath"].as<std::string>(), "quaternion",
                    node["baseName"].as<std::string>(), 
                    node["contact3DofNames"].as<std::vector<std::string>>(), 
                    node["contact6DofNames"].as<std::vector<std::string>>(),
                    node["hipNames"].as<std::vector<std::string>>(),
                    node["verbose"].as<bool>());
    Eigen::VectorXd qj_max(12), qj_min(12);
    for (size_t i=0; i<4; ++i) {
        qj_min.segment(3*i, 3) = yamlToEigenVector(node["jointLimits"]["min"]);
        qj_max.segment(3*i, 3) = yamlToEigenVector(node["jointLimits"]["max"]);
    }
    this->setJointLimits(qj_max, qj_min);
}


void LeggedModel::loadUrdf(std::string urdfPath, std::string baseType, std::string baseName,
                           std::vector<std::string> contact3DofNames, 
                           std::vector<std::string> contact6DofNames, 
                           std::vector<std::string> hipNames, 
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

    hipNames_ = hipNames;
    for(const auto& hipName : hipNames_) hipIds_.push_back(model_.getJointId(hipName));

    // Translation bounds
    model_.lowerPositionLimit.head<3>().setConstant(-10.0);  // x, y, z
    model_.upperPositionLimit.head<3>().setConstant(10.0);

    // Orientation（四元数不设置限制，EulerZYX可以设置为 -pi 到 pi）
    if (baseType_ == "eulerZYX") {
        model_.lowerPositionLimit.segment<3>(3).setConstant(-M_PI);
        model_.upperPositionLimit.segment<3>(3).setConstant(M_PI);
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


Eigen::MatrixXd LeggedModel::jacobian3Dof(Eigen::VectorXd q_pin){
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::computeJointJacobians(model_, data_);
    Eigen::MatrixXd jac(3*nContacts3Dof_, model_.nv);
    for (size_t i = 0; i < nContacts3Dof_; ++i) {
        Eigen::MatrixXd jac_temp(6, model_.nv);
        jac_temp.setZero();
        pinocchio::getFrameJacobian(model_, data_, contact3DofIds_[i], pinocchio::LOCAL_WORLD_ALIGNED, jac_temp);
        jac.block(3*i, 0, 3, model_.nv) = jac_temp.topRows<3>();
    }
    return jac;
}

bool LeggedModel::inverseKine3Dof(Eigen::VectorXd qBase, Eigen::VectorXd& qJoints, std::vector<Eigen::Vector3d> contact3DofPoss){
    if (qBase.size() != nqBase_) {
        throw std::runtime_error("Base pose vector size does not match nqBase_");
    }
    
    if (contact3DofPoss.empty()) {
        Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
        q_pin.head(nqBase_) = qBase;
        pinocchio::forwardKinematics(model_, data_, q_pin);
        for (size_t i = 0; i < contact3DofNames_.size(); ++i) {
            Eigen::Vector3d hip_world = data_.oMi[hipIds_[i]].translation();
            contact3DofPoss.push_back(Eigen::Vector3d(hip_world.x(), hip_world.y(), 0));
        }
        if (verbose_)
            std::cout << "[LeggedModel] Auto-generated default foot targets from hip projections." << std::endl;
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
    Eigen::VectorXd q_max = model_.upperPositionLimit;
    Eigen::VectorXd q_min = model_.lowerPositionLimit;
    Eigen::VectorXd q = (q_min + q_max)/2;
    if (verbose_) std::cout << "[LeggedModel] IK start from " << q.transpose() << std::endl;
    // err = [err_foot_1^T, err_foot_2^T, ...]^T
    Eigen::VectorXd err = Eigen::VectorXd::Zero(nContacts3Dof_*3);
    Eigen::VectorXd dqj = Eigen::VectorXd::Zero(model_.nv-6);
    Eigen::MatrixXd Jj = Eigen::MatrixXd::Zero(nContacts3Dof_*3, model_.nv-6);
    for (int i = 0; i < max_iters; i++) {
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);

        err = desEEpos;
        for (size_t i = 0; i < contact3DofIds_.size(); i++) {
            err.segment(3*i, 3) -= data_.oMf[contact3DofIds_[i]].translation();
        }

        if (err.norm() < tol) {
            if (verbose_) std::cout << "[LeggedModel] IK Converged in " << i << " iterations. Final error: " << err.norm() << std::endl;
            qJoints = q.tail(nJoints_);
            Eigen::VectorXd qj_max = q_max.tail(nJoints_);
            Eigen::VectorXd qj_min = q_min.tail(nJoints_);

            if ( ((qJoints.array() < qj_min.array()) || (qJoints.array() > qj_max.array())).any() ) {
                std::cout << "[LeggedModel] inverseKine3Dof: joint pos out of range." << std::endl;
                qJoints = qJoints.cwiseMax(qj_min).cwiseMin(qj_max);
                return false;
            }

            return true;
        }
        
        Jj = jacobian3Dof(q).rightCols(nJoints_);
        dqj = pseudoInverseDLS(Jj)*err;
        
        q.tail(dqj.size()) += dqj * dt;

        // 将角度包裹到 [-pi, pi]
        for (int j = 0; j < dqj.size(); ++j) {
            double& angle = q[nqBase_ + j];
            angle = std::atan2(std::sin(angle), std::cos(angle));
        }
    }
    return false;
}

// \dot{q}_j = J_j^+(v - J_b \dot{q}_b)
Eigen::VectorXd LeggedModel::inverseDiffKine3Dof(Eigen::VectorXd q_pin, Eigen::VectorXd vBase, std::vector<Eigen::Vector3d> contact3DofVels){
    Eigen::VectorXd desEEvel(nContacts3Dof_ * 3);

    if (contact3DofVels.empty()) {
        for (size_t i = 0; i < contact3DofNames_.size(); ++i) {
            contact3DofVels.push_back(Eigen::Vector3d::Zero());
        }
    }

    for (size_t i = 0; i < contact3DofVels.size(); i++) {
        desEEvel.segment(3*i, 3) = contact3DofVels[i];
    }

    auto J = jacobian3Dof(q_pin);
    auto Jb = J.leftCols(6);
    auto Jj = J.rightCols(nJoints_);

    Eigen::VectorXd v_pin(model_.nv);
    v_pin << vBase, pseudoInverseDLS(Jj)*(desEEvel - Jb*vBase);
    return v_pin;
}

