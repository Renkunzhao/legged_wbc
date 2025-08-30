#include "legged_wbc/LeggedModel.h"

#include <string>

int main(int argc, char **argv)
{
    LeggedModel leggedModel;
    leggedModel.loadUrdf("/tmp/legged_control/go1.urdf", "eulerZYX", "base", {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"}, {});

    std::cout << "[LeggedModel]: " << "nDof " << leggedModel.nDof() << std::endl;
    std::cout << "[LeggedModel]: " << "nJoints " << leggedModel.nJoints() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts3Dof " << leggedModel.nContacts3Dof() << std::endl;
    std::cout << "[LeggedModel]: " << "nContacts6Dof " << leggedModel.nContacts6Dof() << std::endl;

    for(size_t i=0;i<leggedModel.nContacts3Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact3DofNames()[i] << ": " << leggedModel.contact3DofIds()[i] << std::endl;
    for(size_t i=0;i<leggedModel.nContacts6Dof();i++) std::cout << "[LeggedModel]: " << leggedModel.contact6DofNames()[i] << ": " << leggedModel.contact6DofIds()[i] << std::endl;

}
