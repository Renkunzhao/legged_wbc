#ifndef LEGGEDMODEL_H
#define LEGGEDMODEL_H

#include "legged_wbc/LeggedState.h"

#include <cstddef>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>

/**
    * @brief LeggedModel 类，封装了 Pinocchio 模型的基本操作
    * @note baseType_ = "quaternion" 时
                q_pinocchio = [base_pos, base_quaternion(x y z w), q_joint]
                v_pinocchio = [base_linearVel(base), base_angularVel(base), dq_joint]
    * @note baseType_ = "eulerZYX" 时
                q_pinocchio = [base_pos, base_eulerZYX, q_joint]
                v_pinocchio = [base_linearVel(world), base_eulerZYX_dot, dq_joint]
    * @note 使用 pinocchio::rpy 进行旋转变换，需注意 eulerZYX = [yaw pitch roll] = rpy.reverse()
 */
class LeggedModel {
private:
    bool verbose_;

    std::string baseType_;
    pinocchio::Model model_;
    pinocchio::Data data_;

    size_t nJoints_;
    std::vector<std::string> jointNames_;

    size_t nqBase_;
    std::string baseName_;               // 基座名称

    // 3 Dof end effector
    size_t nContacts3Dof_;
    std::vector<std::string> contact3DofNames_;
    std::vector<size_t> contact3DofIds_;

    // 6 Dof end effector
    size_t nContacts6Dof_;
    std::vector<std::string> contact6DofNames_;
    std::vector<size_t> contact6DofIds_;

    std::vector<std::string> hipNames_;
    std::vector<size_t> hipIds_;

public:
    void setVerbose() {verbose_ = true;}
    void unsetVerbose() {verbose_ = false;}

    const std::string& baseType() const {return baseType_;}
    const pinocchio::Model& model() const {return model_;}
    pinocchio::Data& data() {return data_;}

    size_t nDof() const {return  nJoints_ + 6;}
    size_t nJoints() const {return  nJoints_;}
    const std::vector<std::string>& jointNames() const {return jointNames_;}

    size_t nqBase() const {return  nqBase_;}

    size_t nContacts3Dof() const {return  nContacts3Dof_;}
    const std::vector<std::string>& contact3DofNames() const {return  contact3DofNames_;}
    const std::vector<size_t>& contact3DofIds() const {return  contact3DofIds_;}
    std::vector<Eigen::Vector3d> contact3DofPoss(const Eigen::VectorXd& q_pin);
    std::vector<Eigen::Vector3d> contact3DofVels(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin);
    
    size_t nContacts6Dof() const {return  nContacts6Dof_;}
    const std::vector<std::string>& contact6DofNames() const {return  contact6DofNames_;}
    const std::vector<size_t>& contact6DofIds() const {return  contact6DofIds_;}
    std::vector<Eigen::Vector3d> contact6DofPoss(const Eigen::VectorXd& q_pin);
    std::vector<Eigen::Vector3d> contact6DofVels(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin);

    std::vector<Eigen::Vector3d> hipPoss(const Eigen::VectorXd& qBase);
    std::vector<Eigen::Vector3d> hipPossProjected(const Eigen::VectorXd& qBase);
    
    Eigen::Vector3d com(const Eigen::VectorXd& q_pin) {return pinocchio::centerOfMass(model_, data_, q_pin);}
    Eigen::Vector3d vcom(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin) {
        pinocchio::centerOfMass(model_, data_, q_pin, v_pin);
        return data_.vcom[0];
    }
    Eigen::VectorXd hcom(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin) {
        pinocchio::computeCentroidalMomentum(model_, data_, q_pin, v_pin);
        return data_.hg.toVector();
    }

    /*
        stack jacobian of all 3Dof contact point, 3*nContacts3Dof_, nDof
    */
    Eigen::MatrixXd jacobian3Dof(Eigen::VectorXd q_pin);

    Eigen::VectorXd inverseKine3Dof(Eigen::VectorXd qBase, VectorXd qJoints0 = VectorXd(), std::vector<Eigen::Vector3d> contact3DofPoss = {}) {
        Eigen::VectorXd q_pin(nqBase_ + nJoints_), qJoints(nJoints_);
        inverseKine3Dof(qBase, qJoints, qJoints0, contact3DofPoss);
        q_pin << qBase, qJoints;
        return q_pin;
    }
    bool inverseKine3Dof(VectorXd qBase, VectorXd& qJoints, VectorXd qJoints0 = VectorXd(), vector<Vector3d> contact3DofPoss = {});
    Eigen::VectorXd inverseDiffKine3Dof(Eigen::VectorXd q_pin, Eigen::VectorXd vBase, std::vector<Eigen::Vector3d> contact3DofVels = {});

    // Dynamics
    Eigen::VectorXd g(const Eigen::VectorXd& q_pin) {
        return pinocchio::computeGeneralizedGravity(model_, data_, q_pin);
    };

    Eigen::VectorXd nle(const Eigen::VectorXd& q_pin, const Eigen::VectorXd& v_pin) {
        return pinocchio::nonLinearEffects(model_, data_, q_pin, v_pin);
    };

    void loadConfig(const YAML::Node& node);
    void loadUrdf(std::string urdfPath, std::string baseType, std::string baseName, 
        std::vector<std::string> contact3DofNames = {},
        std::vector<std::string> contact6DofNames = {},
        std::vector<std::string> hipNames = {}, 
        bool verbose = false);

    void setJointLimits(Eigen::VectorXd qj_max, Eigen::VectorXd qj_min){
        if (qj_max.size() != nJoints_ || qj_min.size() != nJoints_) {
            throw std::runtime_error("[LeggedModel] setJointLimits: qMax/qMin vector size does not match nJoints_");
        }

        for(size_t i=0;i<nJoints_;++i){
            model_.lowerPositionLimit[nqBase_ + i] = qj_min[i];
            model_.upperPositionLimit[nqBase_ + i] = qj_max[i];
        }
    }

    // This function call createCustomState of leggedState to create a custom state (consistent with q,v used in LeggedModel) in leggedState 
    void creatPinoState(LeggedState& leggedState) const {
        // creat q_pinocchio and v_pinocchio
        if (baseType_ == "quaternion") {
            leggedState.createCustomState("q_pin", {"base_pos", "base_quat", "joint_pos"}, jointNames_);
            leggedState.createCustomState("v_pin", {"base_lin_vel_B", "base_ang_vel_B", "joint_vel"}, jointNames_);
            leggedState.createCustomState("f_pin", {"ee3Dof_fc", "ee6Dof_fc"}, contact3DofNames_, contact6DofNames_);
        } else if (baseType_ == "eulerZYX") {
            leggedState.createCustomState("q_pin", {"base_pos", "base_eulerZYX", "joint_pos"}, jointNames_);
            leggedState.createCustomState("v_pin", {"base_lin_vel_W", "base_eulerZYX_dot", "joint_vel"}, jointNames_);
            leggedState.createCustomState("f_pin", {"ee3Dof_fc", "ee6Dof_fc"}, contact3DofNames_, contact6DofNames_);
        }
    }
};

#endif // LEGGEDMODEL_H
