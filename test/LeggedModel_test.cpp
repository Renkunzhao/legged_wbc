#include "legged_wbc/Yaml.h"
#include "legged_wbc/LeggedModel.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <pinocchio/algorithm/joint-configuration.hpp>

int main(int argc, char **argv)
{
    std::string configFile;
    if (argc!=2) throw std::runtime_error("[LeggedModel_test] configFile path required.");
    else configFile = argv[1];
    std::cout << "[LeggedModel_test] Load config from " << configFile << std::endl;

    YAML::Node configNode = YAML::LoadFile(configFile);

    LeggedModel leggedModel;
    leggedModel.loadConfig(configNode);

    std::cout << "[LeggedModel]: " << "nDof " << leggedModel.nDof() << std::endl;
    std::cout << "[LeggedModel]: " << "nJoints " << leggedModel.nJoints() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts3Dof " << leggedModel.nContacts3Dof() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts6Dof " << leggedModel.nContacts6Dof() << std::endl;

    for(size_t i=0;i<leggedModel.nContacts3Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact3DofNames()[i] << ": " << leggedModel.contact3DofIds()[i] << std::endl;
    for(size_t i=0;i<leggedModel.nContacts6Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact6DofNames()[i] << ": " << leggedModel.contact6DofIds()[i] << std::endl;

    std::cout << "--- Position limits ---\n" 
                << leggedModel.model().lowerPositionLimit.transpose() << "\n" 
                << leggedModel.model().upperPositionLimit.transpose() << std::endl;

    Eigen::VectorXd q_rand = pinocchio::randomConfiguration(leggedModel.model());
    auto contact3DofPoss = leggedModel.contact3DofPoss(q_rand);
    std::cout << "[LeggedModel]: " << "q_rand " << q_rand.transpose() << std::endl;
    for(size_t i=0;i<leggedModel.nContacts3Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact3DofNames()[i] << ": " << contact3DofPoss[i].transpose() << std::endl;

    auto q_ik = leggedModel.inverseKine3Dof(q_rand.head(leggedModel.nqBase()), contact3DofPoss);
    std::cout << "[LeggedModel]: " << "q_ik " << q_ik.transpose() << std::endl;
    std::cout << "[LeggedModel]: " << "err " << (q_rand-q_ik).tail(leggedModel.nJoints()).norm() << std::endl;

    Eigen::VectorXd qBase(7);
    qBase << 0, 0, 0.385, 0, 0, 0, 1;
    Eigen::VectorXd q_pin = leggedModel.inverseKine3Dof(qBase);
    std::cout << "[LeggedModel]: " << "q_pin " << q_pin.transpose() << std::endl;
}
