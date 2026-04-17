#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <LBFGSB.h>
#include <fstream>
#include <sstream>
#include <string>

using namespace Eigen;
using namespace std;

// 使用对齐分配器，避免 Eigen 在 std::vector 中的内存对齐报错
using MatrixList = std::vector<MatrixXd, aligned_allocator<MatrixXd>>;

struct QCQPProblem {
    MatrixXd A;
    MatrixList B;
    int n;
    int m;
};

double dualObjectiveAndGradient(const VectorXd& u, const MatrixXd& A, const MatrixList& B,
                                double gamma, VectorXd& grad);

class DualObjectiveFunctor {
private:
    const MatrixXd& A;
    const MatrixList& B;
    double gamma;

public:
    DualObjectiveFunctor(const MatrixXd& A_, const MatrixList& B_, double gamma_)
        : A(A_), B(B_), gamma(gamma_) {}

    // LBFGSpp 要求的严格接口定义
    double operator()(const VectorXd& u, VectorXd& grad) {
        // 直接复用我们之前写好的底层函数
        return dualObjectiveAndGradient(u, A, B, gamma, grad);
    }
};

// 0.从 CSV 读取矩阵到 Eigen 的辅助函数
MatrixXd readCSV(const std::string& path, int rows, int cols) {
    MatrixXd mat(rows, cols);
    std::ifstream file(path);
    
    if (!file.is_open()) {
        std::cerr << "error：can not open file " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int row = 0;
    while (std::getline(file, line) && row < rows) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        while (std::getline(ss, cell, ',') && col < cols) {
            mat(row, col) = std::stod(cell);
            col++;
        }
        row++;
    }
    return mat;
}

// --- 修改：从 CSV 加载问题的函数 ---
QCQPProblem loadProblemFromCSV(int n, int m) {
    int dim = n + 1;
    
    // 1. 读取 A 矩阵 (dim x dim)
    MatrixXd A = readCSV("../problemdata/A_matrix.csv", dim, dim);
    
    // 2. 读取 B 矩阵展平后的数据 (dim x (dim * (m+1)))
    MatrixXd B_flattened = readCSV("../problemdata/B_matrices.csv", dim, dim * (m + 1));
    
    // 3. 将展平的 B 数据切分回 MatrixList
    MatrixList B(m + 1, MatrixXd(dim, dim));
    for (int i = 0; i < m + 1; ++i) {
        // 从 B_flattened 中提取第 i 个矩阵块
        B[i] = B_flattened.block(0, i * dim, dim, dim);
    }
    
    return {A, B, n, m};
}

// =========================================================================
// 1. 底层数学工具函数 (Math Utils)
// =========================================================================

// 将对称矩阵投影到半正定锥 (PSD)
MatrixXd projectToPSD(MatrixXd A) {
    A = (A + A.transpose()) / 2.0; // 保证对称性
    SelfAdjointEigenSolver<MatrixXd> es(A);
    VectorXd D = es.eigenvalues();
    // 将负特征值截断为 0
    D = D.cwiseMax(0.0);
    return es.eigenvectors() * D.asDiagonal() * es.eigenvectors().transpose();
}

// 检查给定的点是否满足所有不等式约束
bool checkFeasibility(const MatrixList& B_cons, const VectorXd& x) {
    VectorXd eta(x.size() + 1);
    eta << x, 1.0;
    
    for (size_t i = 0; i < B_cons.size(); ++i) {
        double val = eta.transpose() * B_cons[i] * eta;
        if (val < 1.0) {
            return false;
        }
    }
    return true;
}

// =========================================================================
// 2. 对偶问题的目标与梯度评估
// =========================================================================

// 计算对偶目标函数值和梯度
double dualObjectiveAndGradient(const VectorXd& u, const MatrixXd& A, const MatrixList& B, 
                                double gamma, VectorXd& grad) {
    int m_plus_1 = B.size();
    
    // 1. 组装 H 矩阵: H = -A + sum(u_i * B_i)
    MatrixXd H = -A;
    for (int i = 0; i < m_plus_1; ++i) {
        H += u(i) * B[i];
    }
    
    // 2. 投影到 PSD
    MatrixXd H_p = projectToPSD(H);
    
    // 3. 计算目标函数值: gamma/2 * trace(H_p^T * H_p) - sum(u)
    double obj_val = (gamma / 2.0) * (H_p.transpose() * H_p).trace() - u.sum();
    
    // 4. 计算梯度: grad_i = gamma * trace(B_i^T * H_p) - 1
    // trace(B^T * H_p) 等价于 B 和 H_p 的逐元素乘积之和
    for (int i = 0; i < m_plus_1; ++i) {
        grad(i) = gamma * (B[i].array() * H_p.array()).sum() - 1.0;
    }
    
    return obj_val;
}

// =========================================================================
// 3. 射线投影与评估 (Algorithm 1 核心)
// =========================================================================
void projectAndEvaluate(const VectorXd& ksi, const VectorXd& x_hat, 
                        const MatrixXd& A, const MatrixList& B, int m,
                        VectorXd& best_ksi, double& best_f) {
    int n = ksi.size();
    VectorXd eta(n + 1);     eta << ksi, 1.0;
    VectorXd eta_hat(n + 1); eta_hat << x_hat, 1.0;
    VectorXd diff = eta - eta_hat;
    
    double t_max = 0.0;
    double t_min = 0.0;
    bool has_roots = false;
    
    for (int j = 0; j < m; ++j) {
        double a_hat = diff.transpose() * B[j] * diff;
        double b_hat = 2.0 * eta.transpose() * B[j] * diff;
        double c_hat = (eta.transpose() * B[j] * eta)(0,0) - 1.0;
        
        double disc = b_hat * b_hat - 4.0 * a_hat * c_hat;
        if (disc < 0) continue;
        
        double sqrt_disc = std::sqrt(disc);
        double t1 = (-b_hat + sqrt_disc) / (2.0 * a_hat);
        double t2 = (-b_hat - sqrt_disc) / (2.0 * a_hat);
        
        if (!has_roots) {
            t_max = std::max(t1, t2);
            t_min = std::min(t1, t2);
            has_roots = true;
        } else {
            t_max = std::max(t_max, std::max(t1, t2));
            t_min = std::min(t_min, std::min(t1, t2));
        }
    }
    
    VectorXd ksi1 = ksi + t_max * (ksi - x_hat);
    VectorXd ksi2 = ksi + t_min * (ksi - x_hat);
    
    MatrixXd A_sub = A.topLeftCorner(n, n);
    VectorXd A_vec = A.topRightCorner(n, 1);
    double A_const = A(n, n);
    
    double f1 = (ksi1.transpose() * A_sub * ksi1)(0, 0)
              + 2.0 * (A_vec.transpose() * ksi1)(0, 0)
              + A_const;
    double f2 = (ksi2.transpose() * A_sub * ksi2)(0, 0)
              + 2.0 * (A_vec.transpose() * ksi2)(0, 0)
              + A_const;
    
    if (f1 < f2) {
        best_ksi = ksi1;
        best_f = f1;
    } else {
        best_ksi = ksi2;
        best_f = f2;
    }
}

// =========================================================================
// 4. 数据生成器 (对应 MATLAB 的 P3 函数)
// =========================================================================
QCQPProblem generateP3(int n, int m, std::mt19937& gen) {
    std::normal_distribution<double> dist_n(0.0, 1.0);
    std::uniform_real_distribution<double> dist_u(0.0, 1.0);
    
    MatrixXd A_rand = MatrixXd::NullaryExpr(n, n, [&](){ return dist_n(gen); });
    MatrixXd A0 = A_rand * A_rand.transpose();
    VectorXd b0 = VectorXd::NullaryExpr(n, [&](){ return dist_n(gen); });
    
    // c0 = b0' / A0 * b0 + rand; (MATLAB 里的 / 对应乘逆)
    double c0 = (b0.transpose() * A0.llt().solve(b0))(0,0) + dist_u(gen);
    
    MatrixXd A(n + 1, n + 1);
    A.topLeftCorner(n, n) = A0;
    A.topRightCorner(n, 1) = b0;
    A.bottomLeftCorner(1, n) = b0.transpose();
    A(n, n) = c0;
    
    double d = std::pow(n, 3) * A.norm();
    
    MatrixList B(m + 1, MatrixXd(n + 1, n + 1));
    for (int i = 0; i < m; ++i) {
        MatrixXd B_rand = MatrixXd::NullaryExpr(n, n, [&](){ return dist_n(gen); });
        MatrixXd B_i = B_rand * B_rand.transpose();
        VectorXd b_i = VectorXd::NullaryExpr(n, [&](){ return dist_n(gen); });
        double c_i = -dist_u(gen) * d;
        
        B[i].topLeftCorner(n, n) = B_i;
        B[i].topRightCorner(n, 1) = b_i;
        B[i].bottomLeftCorner(1, n) = b_i.transpose();
        B[i](n, n) = c_i;
    }
    
    // 最后一个等式约束
    B[m].setZero();
    B[m](n, n) = 1.0;
    
    return {A, B, n, m};
}

// =========================================================================
// 5. 主流程
// =========================================================================
int main() {
    std::mt19937 gen(3); // 对应 MATLAB 的 rng(3)
    int n = 60;
    int m = 30;
    
    // cout << "=== Initializing QCQP problem (n=" << n << ", m=" << m << ") ===" << endl;
    // QCQPProblem prob = generateP3(n, m, gen);
    // MatrixXd A = prob.A;
    // MatrixList B = prob.B;

    
    cout << "=== load QCQP problem from CSV (n=" << n << ", m=" << m << ") ===" << endl;
    QCQPProblem prob = loadProblemFromCSV(n, m);
    MatrixXd A = prob.A;
    MatrixList B = prob.B;

    auto start_time = chrono::high_resolution_clock::now();
    
    // 步骤 1: 无约束最优解
    VectorXd x_hat = -A.topLeftCorner(n, n).ldlt().solve(A.topRightCorner(n, 1));
    cout << "Unconstrained optimum x_hat: \n" << x_hat << endl;
    
    // 步骤 2: 检查可行性 (只检查前 m 个不等式)
    MatrixList B_ineq(B.begin(), B.begin() + m);
    bool feasible = checkFeasibility(B_ineq, x_hat);
    
    VectorXd best_x = x_hat;
    double min_obj = std::numeric_limits<double>::infinity();
    
    if (feasible) {
        cout << "Unconstrained optimum satisfies all constraints. Returning solution." << endl;
    } else {
        cout << "Constraints violated. Entering SDP solve process..." << endl;
        
        // 步骤 3: 优化对偶问题 (使用LBFGSpp库)
        double gamma = 20.0;
        // 置边界 (Lower Bound & Upper Bound)
        // 对应 MATLAB: lb = [zeros(m,1); -Inf];
        VectorXd lb = VectorXd::Zero(m + 1);
        lb(m) = -std::numeric_limits<double>::infinity(); // 最后一个等式约束是对偶变量无下界
        VectorXd ub = VectorXd::Constant(m + 1, std::numeric_limits<double>::infinity()); // 无上界
        // 初始化对偶变量
        VectorXd u = VectorXd::Ones(m + 1);
        u(m) = 1.0;
        // 配置 L-BFGS-B 参数
        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = 1e-4; // 对应收敛精度，可根据需要调小
        param.max_iterations = 1000;
        // 实例化求解器并开始求解
        LBFGSpp::LBFGSBSolver<double> solver(param);
        DualObjectiveFunctor fun(A, B, gamma);
        cout << "Starting L-BFGS-B optimization..." << endl;
        double final_obj_val;
        try {
            int niter = solver.minimize(fun, u, final_obj_val, lb, ub);
            cout << "L-BFGS-B converged in " << niter << " iterations." << endl;
            cout << "Final Dual Objective: " << final_obj_val << endl;
            // 打印 u 以便和 MATLAB 的 fmincon 结果进行终极对齐测试
            cout << "Optimized u : " << u.transpose() << endl;
        } catch (const std::exception& e) {
            cerr << "L-BFGS-B Optimization failed: " << e.what() << endl;
        }


        // 步骤 4: 恢复 SDP 解矩阵
        MatrixXd H_opt = -A;
        for (int i = 0; i < m + 1; ++i) H_opt += u(i) * B[i];
        MatrixXd X_opt = gamma * projectToPSD(H_opt);
        cout << "X_opt : " << X_opt.transpose() << endl;
        

        // 步骤 5: 多元高斯分布采样与 Cholesky 分解
        int num_samples = 20;
        VectorXd z = X_opt.topRightCorner(n, 1);
        MatrixXd Z = X_opt.topLeftCorner(n, n) - z * z.transpose();
        MatrixXd Z_psd = projectToPSD(Z);
        cout << "Z_psd : " << Z_psd.transpose() << endl;

        // Cholesky 分解 L * L^T = Z_psd
        LLT<MatrixXd> llt(Z_psd);
        MatrixXd L = llt.matrixL();
        cout << "Cholesky factor L : " << L.transpose() << endl;
        
        std::normal_distribution<double> std_norm(0.0, 1.0);

        
        cout << "Starting randomization and ray projection..." << endl;
        for (int i = 0; i < num_samples; ++i) {
            // generate standard normal vector
            VectorXd v(n);
            for(int k=0; k<n; ++k) v(k) = std_norm(gen);
            
            // 转化为目标高斯分布
            VectorXd ksi = z + L * v;
            
            VectorXd ksi_feasible;
            double obj_val;
            
            // 射线投影寻根
            projectAndEvaluate(ksi, x_hat, A, B, m, ksi_feasible, obj_val);
            
            if (obj_val < min_obj) {
                min_obj = obj_val;
                best_x = ksi_feasible;
            }
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;
    
    cout << "\n=== Solve completed ===" << endl;
    cout << "Elapsed time: " << diff.count() * 1000.0 << " ms" << endl;
    cout << "Optimal x: \n" << best_x << endl;
    cout << "Optimal f: \n" << min_obj << endl;
    
    return 0;
}