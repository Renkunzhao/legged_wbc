#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

// 将 YAML list 转换为 Eigen::VectorXd
inline Eigen::VectorXd yamlToEigenVector(const YAML::Node& node) {
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("YAML node is not a valid sequence.");
    }
    std::vector<double> vec = node.as<std::vector<double>>();
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}