#include "legged_wbc/WeightedWbc.h"

#include <cstddef>
#include <iostream>
#include <Eigen/Dense>

int main(int argc, char* argv[]) {
    std::string configFile;
    if (argc!=2) throw std::runtime_error("[WeightedWbc_test] configFile path required.");
    else configFile = argv[1];
    std::cout << "[WeightedWbc_test] Load config from " << configFile << std::endl;

    legged::WeightedWbc weightedWbc;
    weightedWbc.loadTasksSetting(configFile);

    Eigen::VectorXd qMeasured(18), vMeasured(18), qDesired(18), vDesired(18), fDesired(12);
    qMeasured.setZero();
    vMeasured.setZero();
    qDesired.setZero();
    vDesired.setZero();
    fDesired.setZero();

    qMeasured[2] = 0.15;
    qMeasured.tail(12) << -0.5621, 1.3056, -2.5357,
                            -0.5621, 1.3056, -2.5357,
                            0.5621,  1.3056, -2.5357,
                            0.5621,  1.3056, -2.5357;

    qDesired[2] = 0.15;
    vDesired.tail(12) << -0.5621, 1.3056, -2.5357,
                            -0.5621, 1.3056, -2.5357,
                            0.5621,  1.3056, -2.5357,
                            0.5621,  1.3056, -2.5357;

    for(size_t i=0;i<4;i++) fDesired[3*i + 2] = weightedWbc.mass() * 9.81 / 4.0;

    auto x = weightedWbc.update(qDesired, vDesired, fDesired, qMeasured, vMeasured, 
                                        {true, true, true, true}, 0.001, "pd");

    return 0;
}
