#include "legged_wbc/LeggedModel.h"

#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

int main(int argc, char **argv)
{
    std::string configFile;
    if (argc!=2) throw std::runtime_error("[LeggedModel_test] configFile path required.");
    else configFile = argv[1];
    std::cout << "[LeggedModel_test] Load config from " << configFile << std::endl;

    YAML::Node configNode = YAML::LoadFile(configFile);

    LeggedModel leggedModel;
    leggedModel.loadUrdf(configNode["urdfPath"].as<std::string>(), "eulerZYX",
                        configNode["baseName"].as<std::string>(), 
                        configNode["contact3DofNames"].as<std::vector<std::string>>(), 
                        configNode["contact6DofNames"].as<std::vector<std::string>>());

    std::cout << "[LeggedModel]: " << "nDof " << leggedModel.nDof() << std::endl;
    std::cout << "[LeggedModel]: " << "nJoints " << leggedModel.nJoints() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts3Dof " << leggedModel.nContacts3Dof() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts6Dof " << leggedModel.nContacts6Dof() << std::endl;

    for(size_t i=0;i<leggedModel.nContacts3Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact3DofNames()[i] << ": " << leggedModel.contact3DofIds()[i] << std::endl;
    for(size_t i=0;i<leggedModel.nContacts6Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact6DofNames()[i] << ": " << leggedModel.contact6DofIds()[i] << std::endl;

}
