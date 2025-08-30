#ifndef LEGGEDMODEL_H
#define LEGGEDMODEL_H

#include "legged_wbc/LeggedState.h"

#include <cstddef>
#include <string>
#include <vector>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

class LeggedModel {
private:
    std::string baseType_;
    pinocchio::Model model_;
    pinocchio::Data data_;

    size_t nJoints_;
    LeggedState leggedState_;

    std::string baseName_;               // 基座名称

    // 3 Dof end effector
    size_t nContacts3Dof_;
    std::vector<std::string> contact3DofNames_;
    std::vector<size_t> contact3DofIds_;

    // 6 Dof end effector
    size_t nContacts6Dof_;
    std::vector<std::string> contact6DofNames_;
    std::vector<size_t> contact6DofIds_;

    std::vector<std::string> hipNames_;  // 机器人髋关节的名称

public:
    const std::string& baseType() const {return baseType_;}
    const pinocchio::Model& model() const {return model_;}
    pinocchio::Data& data() {return data_;}

    size_t nDof() const {return  nJoints_ + 6;}
    size_t nJoints() const {return  nJoints_;}

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

    void loadUrdf(std::string urdfPath, std::string baseType, std::string baseName, std::vector<std::string> contact3DofNames, std::vector<std::string> contact6DofNames);
};

#endif // LEGGEDMODEL_H
