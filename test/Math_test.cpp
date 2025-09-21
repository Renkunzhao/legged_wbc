#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "legged_wbc/Math.h"

using namespace std;
using namespace std::chrono;

int main() {
    // 矩阵规模
    int m = 60;   // 行数
    int n = 40;   // 列数

    // 随机矩阵和向量
    Eigen::MatrixXd J = Eigen::MatrixXd::Random(m, n);
    Eigen::VectorXd v = Eigen::VectorXd::Random(m);

    // 测试次数
    int N = 1000;

    // ---------------- SVD 测试 ----------------
    auto start = high_resolution_clock::now();
    Eigen::VectorXd dq_svd;
    for (int i = 0; i < N; i++) {
        dq_svd = pseudoInverseSVD(J) * v;   // 直接计算 J^+ v
    }
    auto end = high_resolution_clock::now();
    double time_svd = duration_cast<milliseconds>(end - start).count();

    // ---------------- DLS 测试 ----------------
    start = high_resolution_clock::now();
    Eigen::VectorXd dq_dls;
    for (int i = 0; i < N; i++) {
        dq_dls = pseudoInverseDLS(J) * v;   // 直接计算 J^+_λ v
    }
    end = high_resolution_clock::now();
    double time_dls = duration_cast<milliseconds>(end - start).count();

    // ---------------- 输出结果 ----------------
    cout << "Matrix J: " << m << "x" << n << ", iterations = " << N << endl;
    cout << "SVD   time = " << time_svd << " ms" << endl;
    cout << "DLS   time = " << time_dls << " ms" << endl;
    cout << "||dq_svd - dq_dls|| = " << (dq_svd - dq_dls).norm() << endl;

    return 0;
}
