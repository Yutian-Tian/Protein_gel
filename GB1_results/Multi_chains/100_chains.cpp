// M条N个domain的链的定力系综优化
// 仅考虑能量的分布（结构异质性），链长xi_f为常数

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <string>
#include <map>
#include <set>
#include <filesystem>
#include <iomanip>
#include <sstream>

using namespace std;
namespace fs = filesystem;

// ===================== 全局参数（与Python代码2完全对齐） =====================
const double xi_f = 3.6;       // 折叠态长度，固定常数
const double alpha = 7.6;      // 解折叠长度系数
const double E_mean = 11.9;    // 解折叠能均值
const double E_std = 1.7;      // 解折叠能标准差
const double E_delta = 5.0;    // 能量截断范围
const int N = 1;              // 每条链的domain数量
const int M = 100;             // 链的总数量
const int r_grid = 100;        // r方向网格数
const int n_grid = 10;         // n方向网格数
const int f_grid_initial = 500;// 初始力值网格数
const double f_max = 10.0;     // 最大扫描力值

// 自适应细化参数
const double refinement_threshold = 0.1;
const int max_refinement_level = 10;
const double tolerance = 1e-8;

// 能量采样上下界
const double Upper_bound = E_mean + E_delta;
const double Lower_bound = E_mean - E_delta;

// 随机数生成器（采样ΔE分布）
random_device rd;
mt19937 gen(rd());
normal_distribution<> dist(E_mean, E_std);

// ===================== 数据结构定义 =====================
struct DomainResult {
    int domain_idx;
    double xi_f;
    vector<double> f_vals;
    vector<double> r_opt;
    vector<double> n_opt;
    vector<double> Fd_min;
    vector<double> x_opt;
    double DeltaE;       // 该domain的解折叠能
    int num_points;
};

struct ChainResult {
    int chain_idx;
    vector<double> DeltaE; // 该链所有domain的解折叠能
    vector<double> unified_f_vals;
    vector<vector<double>> unified_r_opt;  // [f_index][domain]
    vector<vector<double>> unified_n_opt;
    vector<vector<double>> unified_Fd_min;
    vector<vector<double>> unified_x_opt;
    vector<DomainResult> all_domain_results;
};

struct OptimizationResult {
    double r_best;
    double n_best;
    double Fd_min;
    double x_best;
};

struct DomainOptimizationResult {
    vector<double> f_vals;
    vector<double> r_opt;
    vector<double> n_opt;
    vector<double> Fd_min;
    vector<double> x_opt;
};

// ===================== 辅助工具函数 =====================
bool isClose(double a, double b, double eps = 1e-10) {
    return abs(a - b) < eps;
}

vector<double> linspace(double start, double stop, int num) {
    vector<double> result(num);
    if (num == 1) {
        result[0] = start;
        return result;
    }
    
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
    return result;
}

// ===================== 物理模型函数 =====================
double energy_term_U(double n_i, double DeltaEi) {
    return DeltaEi * n_i - DeltaEi * cos(2 * M_PI * n_i);
}

double contour_length_Lci(double n_i) {
    double xi_ui = alpha * xi_f;
    return xi_f + n_i * (xi_ui - xi_f);
}

double end_to_end_factor_x_i(double r_i, double n_i) {
    double L_ci = contour_length_Lci(n_i);
    return r_i / L_ci;
}

double WLC_free_energy(double x_i, double L_ci) {
    if (x_i >= 0.999) {
        return numeric_limits<double>::infinity();
    }
    return 0.25 * L_ci * (x_i * x_i * (3.0 - 2.0 * x_i) / (1.0 - x_i));
}

double single_domain_free_energy(double r_i, double n_i, double DeltaEi, double f_ext = 0.0) {
    double L_ci = contour_length_Lci(n_i);
    double x_i = end_to_end_factor_x_i(r_i, n_i);
    double F_wlc = WLC_free_energy(x_i, L_ci);
    double Ui = energy_term_U(n_i, DeltaEi);
    double work_term = f_ext * x_i * L_ci;
    return F_wlc + Ui - work_term;
}

// ===================== 单点优化（网格迭代细化） =====================
OptimizationResult optimize_single_point(double f, double DeltaEi, double opt_tol = 1e-8) {
    double r_max_initial = contour_length_Lci(1.0);
    
    double r_search_min = 0.0;
    double r_search_max = r_max_initial;
    double n_search_min = 0.0;
    double n_search_max = 1.0;
    
    // 初始步长（与Python版对齐，使用num-1计算）
    double r_step = (r_search_max - r_search_min) / (r_grid - 1);
    double n_step = (n_search_max - n_search_min) / (n_grid - 1);
    
    double min_Fd = numeric_limits<double>::infinity();
    double r_best = 0.0, n_best = 0.0;
    
    while (true) {
        vector<double> r_vals = linspace(r_search_min, r_search_max, r_grid);
        vector<double> n_vals = linspace(n_search_min, n_search_max, n_grid);
        
        for (double r : r_vals) {
            for (double n : n_vals) {
                double Fd = single_domain_free_energy(r, n, DeltaEi, f);
                if (Fd < min_Fd) {
                    min_Fd = Fd;
                    r_best = r;
                    n_best = n;
                }
            }
        }
        
        if (r_step <= opt_tol && n_step <= opt_tol) {
            break;
        }
        
        // 缩小搜索范围
        r_search_min = max(0.0, r_best - r_step);
        r_search_max = min(r_max_initial, r_best + r_step);
        n_search_min = max(0.0, n_best - n_step);
        n_search_max = min(1.0, n_best + n_step);
        
        // 更新步长
        r_step = (r_search_max - r_search_min) / 10.0;
        n_step = (n_search_max - n_search_min) / 10.0;
    }
    
    double x_best = end_to_end_factor_x_i(r_best, n_best);
    return {r_best, n_best, min_Fd, x_best};
}

// ===================== 力值自适应细化 =====================
vector<pair<double, double>> detect_transition_regions(
    const vector<double>& f_vals, 
    const vector<double>& n_opt, 
    double threshold = 0.5) {
    
    vector<pair<double, double>> transition_regions;
    int n = f_vals.size();
    
    for (int i = 0; i < n - 1; i++) {
        double dn = n_opt[i + 1] - n_opt[i];
        if (abs(dn) >= threshold) {
            transition_regions.emplace_back(f_vals[i], f_vals[i + 1]);
        }
    }
    return transition_regions;
}

vector<double> refine_force_grid(
    const vector<double>& f_vals,
    const vector<double>& n_opt,
    double refinement_threshold = 0.5,
    double tol = 1e-10) {
    
    set<double> refined_f(f_vals.begin(), f_vals.end());
    vector<pair<double, double>> transition_regions = 
        detect_transition_regions(f_vals, n_opt, refinement_threshold);
    
    for (const auto& region : transition_regions) {
        double f_mid = (region.first + region.second) / 2.0;
        
        bool exists = false;
        for (double f : refined_f) {
            if (abs(f_mid - f) < tol) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            refined_f.insert(f_mid);
        }
    }
    return vector<double>(refined_f.begin(), refined_f.end());
}

// ===================== 单个domain自适应优化 =====================
DomainOptimizationResult optimize_single_domain_adaptive(double DeltaEi) {
    vector<double> f_vals_current = linspace(0.0, f_max, f_grid_initial);
    
    vector<double> r_opt_current, n_opt_current, Fd_min_current, x_opt_current;
    
    // 初始网格全量计算
    for (double f : f_vals_current) {
        auto result = optimize_single_point(f, DeltaEi);
        r_opt_current.push_back(result.r_best);
        n_opt_current.push_back(result.n_best);
        Fd_min_current.push_back(result.Fd_min);
        x_opt_current.push_back(result.x_best);
    }
    
    // 自适应细化循环
    for (int level = 0; level < max_refinement_level; level++) {
        vector<double> refined_f_vals = refine_force_grid(
            f_vals_current, n_opt_current, refinement_threshold);
        
        if (refined_f_vals.size() == f_vals_current.size()) {
            break;
        }
        
        // 检查最小力值间隔
        if (refined_f_vals.size() > 1) {
            double min_f_interval = numeric_limits<double>::infinity();
            for (size_t i = 1; i < refined_f_vals.size(); i++) {
                double diff = refined_f_vals[i] - refined_f_vals[i - 1];
                if (diff < min_f_interval) {
                    min_f_interval = diff;
                }
            }
            if (min_f_interval <= tolerance) {
                break;
            }
        }
        
        // 筛选新增力值点
        vector<double> new_f_points;
        for (double f : refined_f_vals) {
            bool is_new = true;
            for (double existing_f : f_vals_current) {
                if (abs(f - existing_f) < 1e-10) {
                    is_new = false;
                    break;
                }
            }
            if (is_new) {
                new_f_points.push_back(f);
            }
        }
        
        if (new_f_points.empty()) {
            break;
        }
        
        // 仅计算新增点
        vector<double> new_r_opt, new_n_opt, new_Fd_min, new_x_opt;
        for (double f : new_f_points) {
            auto result = optimize_single_point(f, DeltaEi);
            new_r_opt.push_back(result.r_best);
            new_n_opt.push_back(result.n_best);
            new_Fd_min.push_back(result.Fd_min);
            new_x_opt.push_back(result.x_best);
        }
        
        // 合并新旧结果
        map<double, vector<double>> result_dict;
        for (size_t i = 0; i < f_vals_current.size(); i++) {
            result_dict[f_vals_current[i]] = {
                r_opt_current[i], n_opt_current[i], Fd_min_current[i], x_opt_current[i]
            };
        }
        for (size_t i = 0; i < new_f_points.size(); i++) {
            result_dict[new_f_points[i]] = {
                new_r_opt[i], new_n_opt[i], new_Fd_min[i], new_x_opt[i]
            };
        }
        
        // 按力值排序重建数组
        vector<double> f_vals_final = refined_f_vals;
        vector<double> r_opt_final, n_opt_final, Fd_min_final, x_opt_final;
        for (double f : f_vals_final) {
            const auto& res = result_dict[f];
            r_opt_final.push_back(res[0]);
            n_opt_final.push_back(res[1]);
            Fd_min_final.push_back(res[2]);
            x_opt_final.push_back(res[3]);
        }
        
        f_vals_current = f_vals_final;
        r_opt_current = r_opt_final;
        n_opt_current = n_opt_final;
        Fd_min_current = Fd_min_final;
        x_opt_current = x_opt_final;
        
        // 二次检查精度
        if (f_vals_current.size() > 1) {
            double min_f_interval = numeric_limits<double>::infinity();
            for (size_t i = 1; i < f_vals_current.size(); i++) {
                double diff = f_vals_current[i] - f_vals_current[i - 1];
                if (diff < min_f_interval) {
                    min_f_interval = diff;
                }
            }
            if (min_f_interval <= tolerance) {
                break;
            }
        }
    }
    
    return {f_vals_current, r_opt_current, n_opt_current, Fd_min_current, x_opt_current};
}

// ===================== 生成单条链的ΔE分布 =====================
vector<double> generate_DeltaE(int n) {
    vector<double> DeltaE_samples;
    
    while (DeltaE_samples.size() < n) {
        double sample = dist(gen);
        if (sample >= Lower_bound && sample <= Upper_bound) {
            DeltaE_samples.push_back(sample);
        }
    }
    return DeltaE_samples;
}

// ===================== 单条链全流程优化 =====================
ChainResult optimize_single_chain(int chain_idx) {
    cout << "\n开始优化链 " << chain_idx + 1 << "/" << M << "...\n";
    
    // 生成该链所有domain的解折叠能
    vector<double> chain_DeltaE = generate_DeltaE(N);
    vector<DomainResult> all_domain_results;
    
    // 逐个domain自适应优化
    for (int i = 0; i < N; i++) {
        DomainOptimizationResult domain_opt = optimize_single_domain_adaptive(chain_DeltaE[i]);
        
        DomainResult domain_result;
        domain_result.domain_idx = i;
        domain_result.xi_f = xi_f;
        domain_result.f_vals = domain_opt.f_vals;
        domain_result.r_opt = domain_opt.r_opt;
        domain_result.n_opt = domain_opt.n_opt;
        domain_result.Fd_min = domain_opt.Fd_min;
        domain_result.x_opt = domain_opt.x_opt;
        domain_result.DeltaE = chain_DeltaE[i];
        domain_result.num_points = domain_opt.f_vals.size();
        
        all_domain_results.push_back(domain_result);
    }
    
    // 构建统一力值网格（所有domain力值点的并集）
    set<double> all_f_vals_set;
    for (const auto& result : all_domain_results) {
        all_f_vals_set.insert(result.f_vals.begin(), result.f_vals.end());
    }
    vector<double> unified_f_vals(all_f_vals_set.begin(), all_f_vals_set.end());
    sort(unified_f_vals.begin(), unified_f_vals.end());
    
    cout << "  链 " << chain_idx + 1 << " 统一力值网格点数: " << unified_f_vals.size() << "\n";
    cout << "  链 " << chain_idx + 1 << " 在统一网格上重新优化所有domain...\n";
    
    // 在统一网格上重新计算所有domain
    vector<vector<double>> unified_r_opt(unified_f_vals.size(), vector<double>(N));
    vector<vector<double>> unified_n_opt(unified_f_vals.size(), vector<double>(N));
    vector<vector<double>> unified_Fd_min(unified_f_vals.size(), vector<double>(N));
    vector<vector<double>> unified_x_opt(unified_f_vals.size(), vector<double>(N));
    
    for (int i = 0; i < N; i++) {
        for (size_t j = 0; j < unified_f_vals.size(); j++) {
            auto result = optimize_single_point(unified_f_vals[j], chain_DeltaE[i]);
            unified_r_opt[j][i] = result.r_best;
            unified_n_opt[j][i] = result.n_best;
            unified_Fd_min[j][i] = result.Fd_min;
            unified_x_opt[j][i] = result.x_best;
        }
    }
    
    cout << "  链 " << chain_idx + 1 << " 优化完成\n";
    
    ChainResult chain_result;
    chain_result.chain_idx = chain_idx;
    chain_result.DeltaE = chain_DeltaE;
    chain_result.unified_f_vals = unified_f_vals;
    chain_result.unified_r_opt = unified_r_opt;
    chain_result.unified_n_opt = unified_n_opt;
    chain_result.unified_Fd_min = unified_Fd_min;
    chain_result.unified_x_opt = unified_x_opt;
    chain_result.all_domain_results = all_domain_results;
    
    return chain_result;
}

// ===================== 文件保存函数 =====================
void save_vector_to_csv(const vector<double>& data, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << "\n";
        return;
    }
    
    file << scientific << setprecision(15);
    for (size_t i = 0; i < data.size(); i++) {
        file << data[i];
        if (i < data.size() - 1) file << ",";
    }
    file << "\n";
}

void save_matrix_to_csv(const vector<vector<double>>& data, 
                       const vector<double>& f_vals,
                       const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << "\n";
        return;
    }
    
    file << scientific << setprecision(15);
    
    // 列标题
    file << ",";
    for (int i = 1; i <= N; ++i) {
        file << "Domain_" << i;
        if (i < N) file << ",";
    }
    file << "\n";
    
    // 数据行
    for (size_t j = 0; j < f_vals.size(); ++j) {
        file << f_vals[j];
        for (int i = 0; i < N; ++i) {
            file << "," << data[j][i];
        }
        file << "\n";
    }
}

void save_chain_results(const ChainResult& chain_result, const string& save_path) {
    int chain_idx = chain_result.chain_idx;
    string prefix = save_path + "/chain_" + to_string(chain_idx + 1);
    
    save_matrix_to_csv(chain_result.unified_r_opt, chain_result.unified_f_vals,
                      prefix + "_r_values_unified.csv");
    save_matrix_to_csv(chain_result.unified_n_opt, chain_result.unified_f_vals,
                      prefix + "_n_values_unified.csv");
    save_matrix_to_csv(chain_result.unified_Fd_min, chain_result.unified_f_vals,
                      prefix + "_Fd_values_unified.csv");
    save_matrix_to_csv(chain_result.unified_x_opt, chain_result.unified_f_vals,
                      prefix + "_x_values_unified.csv");
}

void save_all_chains_DeltaE(const vector<vector<double>>& all_chain_DeltaE, 
                           const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << "\n";
        return;
    }
    
    file << scientific << setprecision(15);
    
    // 列标题
    file << ",";
    for (int j = 1; j <= N; ++j) {
        file << "Domain_" << j;
        if (j < N) file << ",";
    }
    file << "\n";
    
    // 数据行
    for (int i = 0; i < M; ++i) {
        file << "Chain_" << (i+1);
        const auto& chain = all_chain_DeltaE[i];
        for (int j = 0; j < N; ++j) {
            file << "," << chain[j];
        }
        file << "\n";
    }
}

// ===================== 主函数 =====================
int main() {
    string save_path = "/home/tyt/project/protein_gel/GB1_results/Multi_chains/N_" + to_string(N) + "_results";
    
    if (!fs::exists(save_path)) {
        fs::create_directories(save_path);
    }
    
    cout << "==============================================================\n";
    cout << "开始模拟 " << M << " 条链，每条链包含 " << N << " 个domain\n";
    cout << "ξ_f: 固定值 = " << xi_f << "\n";
    cout << "ΔE分布: 均值=" << E_mean << ", 标准差=" << E_std 
              << ", 截断范围=[" << Lower_bound << ", " << Upper_bound << "]\n";
    cout << "力值范围: [0, " << f_max << "]\n";
    cout << "保存路径: " << save_path << "\n";
    cout << "==============================================================\n";
    
    vector<vector<double>> all_chain_DeltaE(M, vector<double>(N));
    vector<ChainResult> all_chain_results;
    
    int progress_interval = max(1, M / 20);
    
    // 优化所有链
    for (int chain_idx = 0; chain_idx < M; chain_idx++) {
        if ((chain_idx + 1) % progress_interval == 0 || (chain_idx + 1) == M) {
            cout << "\n进度: " << chain_idx + 1 << "/" << M << " 条链 ("
                      << ((chain_idx + 1.0) / M * 100.0) << "%)\n";
            cout.flush();
        }
        
        ChainResult chain_result = optimize_single_chain(chain_idx);
        all_chain_DeltaE[chain_idx] = chain_result.DeltaE;
        save_chain_results(chain_result, save_path);
        all_chain_results.push_back(chain_result);
    }
    
    // 保存所有链的ΔE参数
    cout << "\n保存所有链的ΔE参数...\n";
    save_all_chains_DeltaE(all_chain_DeltaE, save_path + "/all_chains_DeltaE.csv");
    
    // 保存模拟参数
    cout << "保存模拟参数...\n";
    ofstream params_file(save_path + "/simulation_parameters.csv");
    if (params_file.is_open()) {
        params_file << "Parameter,Value,Description\n";
        params_file << "xi_f," << xi_f << ",折叠态长度（固定值）\n";
        params_file << "alpha," << alpha << ",解折叠长度系数\n";
        params_file << "E_mean," << E_mean << ",解折叠能均值\n";
        params_file << "E_std," << E_std << ",解折叠能标准差\n";
        params_file << "E_delta," << E_delta << ",能量截断半宽\n";
        params_file << "N_domains_per_chain," << N << ",每条链的domain数量\n";
        params_file << "N_chains," << M << ",链的数量\n";
        params_file << "r_grid," << r_grid << ",r方向网格数\n";
        params_file << "n_grid," << n_grid << ",n方向网格数\n";
        params_file << "f_grid_initial," << f_grid_initial << ",初始力值网格数\n";
        params_file << "f_max," << f_max << ",最大扫描力值\n";
        params_file << "refinement_threshold," << refinement_threshold << ",力值细化阈值\n";
        params_file << "max_refinement_level," << max_refinement_level << ",最大细化层级\n";
        params_file << "tolerance," << tolerance << ",最小力值间隔精度\n";
        params_file << "upper_bound," << Upper_bound << ",ΔE采样上界\n";
        params_file << "lower_bound," << Lower_bound << ",ΔE采样下界\n";
        params_file.close();
    }
    
    // 统计信息
    cout << "创建汇总统计信息...\n";
    vector<int> chain_points;
    for (const auto& chain_result : all_chain_results) {
        chain_points.push_back(chain_result.unified_f_vals.size());
    }
    
    double mean_DeltaE = 0.0, min_DeltaE = numeric_limits<double>::infinity(), 
           max_DeltaE = -numeric_limits<double>::infinity();
    for (const auto& chain : all_chain_DeltaE) {
        for (double e : chain) {
            mean_DeltaE += e;
            if (e < min_DeltaE) min_DeltaE = e;
            if (e > max_DeltaE) max_DeltaE = e;
        }
    }
    mean_DeltaE /= (M * N);
    
    ofstream stats_file(save_path + "/simulation_statistics.csv");
    if (stats_file.is_open()) {
        stats_file << "Statistic,Value\n";
        stats_file << "总链数," << M << "\n";
        stats_file << "每条链domain数," << N << "\n";
        
        double sum_points = 0.0;
        int min_points = numeric_limits<int>::max();
        int max_points = 0;
        for (int points : chain_points) {
            sum_points += points;
            if (points < min_points) min_points = points;
            if (points > max_points) max_points = points;
        }
        double mean_points = sum_points / M;
        stats_file << "平均力值点数," << mean_points << "\n";
        stats_file << "最小力值点数," << min_points << "\n";
        stats_file << "最大力值点数," << max_points << "\n";
        
        double std_points = 0.0;
        for (int points : chain_points) {
            std_points += (points - mean_points) * (points - mean_points);
        }
        std_points = sqrt(std_points / M);
        stats_file << "力值点数标准差," << std_points << "\n";
        
        stats_file << "平均ΔE," << mean_DeltaE << "\n";
        
        double std_DeltaE = 0.0;
        for (const auto& chain : all_chain_DeltaE) {
            for (double e : chain) {
                std_DeltaE += (e - mean_DeltaE) * (e - mean_DeltaE);
            }
        }
        std_DeltaE = sqrt(std_DeltaE / (M * N));
        stats_file << "ΔE标准差," << std_DeltaE << "\n";
        stats_file << "最小ΔE," << min_DeltaE << "\n";
        stats_file << "最大ΔE," << max_DeltaE << "\n";
        
        stats_file.close();
    }
    
    cout << "\n==============================================================\n";
    cout << "模拟完成！\n";
    cout << "==============================================================\n";
    cout << "\n结果已保存到 '" << save_path << "' 目录\n";
    cout << "\n生成的文件:\n";
    cout << "  主要数据文件 (每条链):\n";
    cout << "    - chain_X_r_values_unified.csv: 第X条链的最优端到端距离 (X=1-" << M << ")\n";
    cout << "    - chain_X_n_values_unified.csv: 第X条链的最优展开分数\n";
    cout << "    - chain_X_Fd_values_unified.csv: 第X条链的最小自由能\n";
    cout << "    - chain_X_x_values_unified.csv: 第X条链的最优端到端因子\n";
    cout << "  汇总文件:\n";
    cout << "    - all_chains_DeltaE.csv: 所有链的ΔE参数 (M×N矩阵)\n";
    cout << "    - simulation_parameters.csv: 模拟参数\n";
    cout << "    - simulation_statistics.csv: 模拟统计信息\n";
    cout << "\n数据格式说明:\n";
    cout << "  - 每个CSV文件的第一行是列标题：第一列为空，然后是 Domain_1 到 Domain_" << N << "\n";
    cout << "  - 后续每行：第一列是力值（或链标签），后面是相应值\n";
    cout << "  - all_chains_DeltaE.csv: 第一列是链标签 Chain_1...Chain_" << M << "\n";
    
    return 0;
}