// 并行计算版本的多链优化
// 编译命令：g++ -fopenmp -O3 -o <process_name> Multi_chains_pall.cpp

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
#include <tuple>
#include <functional>
#include <omp.h>   // OpenMP 并行

using namespace std;
namespace fs = std::filesystem;

// ===================== 全局参数（不可变） =====================
struct SimParams {
    double xi_f = 3.6;
    double alpha = 7.6;
    double E_mean = 11.9;
    double E_std = 1.7;
    double E_delta = 5.0;
    int N_domains = 1;      // 每条链domain数
    int M_chains = 300;      // 链总数
    int n_grid = 20;         // n方向初始网格点数
    int f_grid_initial = 5000;
    double f_max = 10.0;
    double refinement_threshold = 0.05;
    int max_refinement_level = 10;
    double tolerance = 1e-8;

    double upper_bound() const { return E_mean + E_delta; }
    double lower_bound() const { return E_mean - E_delta; }
};

// ===================== 结果数据结构 =====================
struct OptResult {
    double r_best;
    double n_best;
    double Fd_min;
    double x_best;
};

struct DomainOptResult {
    vector<double> f_vals;
    vector<double> r_opt;
    vector<double> n_opt;
    vector<double> Fd_min;
    vector<double> x_opt;
};

struct DomainResult {
    int domain_idx;
    double DeltaE;
    DomainOptResult adaptive_result;
};

struct ChainResult {
    int chain_idx;
    vector<double> DeltaE;
    vector<double> unified_f_vals;
    vector<vector<double>> unified_r_opt;   // [domain][f_index]
    vector<vector<double>> unified_n_opt;
    vector<vector<double>> unified_Fd_min;
    vector<vector<double>> unified_x_opt;
    vector<DomainResult> domain_results;
};

// ===================== 辅助工具 =====================
vector<double> linspace(double start, double stop, int num) {
    vector<double> result(num);
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) result[i] = start + i * step;
    return result;
}

// ===================== 物理模型函数 =====================
double energy_term_U(double n, double DeltaE) {
    return DeltaE * n - DeltaE * cos(2.0 * M_PI * n);
}

double contour_length_Lci(double n, const SimParams& p) {
    double xi_u = p.alpha * p.xi_f;
    return p.xi_f + n * (xi_u - p.xi_f);
}

double end_to_end_factor_x(double r, double n, const SimParams& p) {
    return r / contour_length_Lci(n, p);
}

double WLC_free_energy(double x, double L) {
    if (x >= 0.999) return numeric_limits<double>::infinity();
    return 0.25 * L * (x * x * (3.0 - 2.0 * x) / (1.0 - x));
}

double MSforce(double x) {
    if (x >= 0.99) return numeric_limits<double>::infinity();
    double inv = 1.0 / (1.0 - x);
    return 0.25 * (inv * inv - 1.0 + 4.0 * x);
}

double single_domain_free_energy(double x, double n, double DeltaE, double f_ext, const SimParams& p) {
    double L = contour_length_Lci(n, p);
    double F_wlc = WLC_free_energy(x, L);
    double U = energy_term_U(n, DeltaE);
    double work = f_ext * x * L;
    return F_wlc + U - work;
}

// ===================== 二分法求解 x(f) =====================
double solve_x_from_f(double f, double tol = 1e-12) {
    if (f <= 0.0) return 0.0;
    double lo = 0.0, hi = 1.0 - 1e-15;
    while (MSforce(hi) < f) {
        hi = 1.0 - (1.0 - hi) * 0.5;
    }
    for (int i = 0; i < 100; ++i) {
        double mid = (lo + hi) / 2.0;
        double fmid = MSforce(mid);
        if (fmid < f) lo = mid;
        else hi = mid;
        if (hi - lo < tol) break;
    }
    return (lo + hi) / 2.0;
}

// ===================== 单点优化（一维自适应 n 搜索） =====================
OptResult optimize_single_point(double f, double DeltaE, const SimParams& p) {
    double x_root = solve_x_from_f(f);

    double n_min = 0.0, n_max = 1.0;
    double n_step = (n_max - n_min) / (p.n_grid - 1);
    double best_n = 0.0, best_F = numeric_limits<double>::infinity();

    while (true) {
        vector<double> n_vals = linspace(n_min, n_max, p.n_grid);
        for (double n : n_vals) {
            double Fd = single_domain_free_energy(x_root, n, DeltaE, f, p);
            if (Fd < best_F) {
                best_F = Fd;
                best_n = n;
            }
        }
        if (n_step <= p.tolerance) break;

        n_min = max(0.0, best_n - n_step);
        n_max = min(1.0, best_n + n_step);
        n_step = (n_max - n_min) / 10.0;
    }

    double best_Lc = contour_length_Lci(best_n, p);
    double best_r = x_root * best_Lc;
    return { best_r, best_n, best_F, x_root };
}

// ===================== 力值自适应细化 =====================
vector<pair<double, double>> detect_transition_regions(
    const vector<double>& f_vals, const vector<double>& n_opt, double threshold)
{
    vector<pair<double, double>> regions;
    for (size_t i = 0; i < f_vals.size() - 1; ++i) {
        if (abs(n_opt[i+1] - n_opt[i]) >= threshold) {
            regions.emplace_back(f_vals[i], f_vals[i+1]);
        }
    }
    return regions;
}

vector<double> refine_force_grid(const vector<double>& f_vals,
                                 const vector<double>& n_opt,
                                 const SimParams& p)
{
    set<double> existing(f_vals.begin(), f_vals.end());
    auto regions = detect_transition_regions(f_vals, n_opt, p.refinement_threshold);
    for (auto& reg : regions) {
        double f_mid = (reg.first + reg.second) / 2.0;
        bool exists = false;
        for (double ef : existing) {
            if (abs(f_mid - ef) < p.tolerance) { exists = true; break; }
        }
        if (!exists) existing.insert(f_mid);
    }
    return vector<double>(existing.begin(), existing.end());
}

// ===================== 单个 domain 自适应力网格优化 =====================
DomainOptResult optimize_single_domain_adaptive(double DeltaE, const SimParams& p) {
    // 初始力网格
    vector<double> f_vals = linspace(0.0, p.f_max, p.f_grid_initial);
    vector<double> r_opt, n_opt, Fd_min, x_opt;
    for (double f : f_vals) {
        auto res = optimize_single_point(f, DeltaE, p);
        r_opt.push_back(res.r_best);
        n_opt.push_back(res.n_best);
        Fd_min.push_back(res.Fd_min);
        x_opt.push_back(res.x_best);
    }

    for (int level = 0; level < p.max_refinement_level; ++level) {
        vector<double> refined_f = refine_force_grid(f_vals, n_opt, p);
        if (refined_f.size() == f_vals.size()) break;

        // 检查最小力值间隔
        if (refined_f.size() > 1) {
            double min_diff = numeric_limits<double>::infinity();
            for (size_t i = 1; i < refined_f.size(); ++i)
                min_diff = min(min_diff, refined_f[i] - refined_f[i-1]);
            if (min_diff <= p.tolerance) break;
        }

        // 筛选新增力值点
        vector<double> new_f_points;
        for (double f : refined_f) {
            bool is_new = true;
            for (double old_f : f_vals)
                if (abs(f - old_f) < 1e-10) { is_new = false; break; }
            if (is_new) new_f_points.push_back(f);
        }
        if (new_f_points.empty()) break;

        // 仅计算新增点
        vector<double> new_r, new_n, new_F, new_x;
        for (double f : new_f_points) {
            auto res = optimize_single_point(f, DeltaE, p);
            new_r.push_back(res.r_best);
            new_n.push_back(res.n_best);
            new_F.push_back(res.Fd_min);
            new_x.push_back(res.x_best);
        }

        // 合并所有结果并排序
        map<double, tuple<double,double,double,double>> all;
        for (size_t i = 0; i < f_vals.size(); ++i)
            all[f_vals[i]] = make_tuple(r_opt[i], n_opt[i], Fd_min[i], x_opt[i]);
        for (size_t i = 0; i < new_f_points.size(); ++i)
            all[new_f_points[i]] = make_tuple(new_r[i], new_n[i], new_F[i], new_x[i]);

        f_vals.clear(); r_opt.clear(); n_opt.clear(); Fd_min.clear(); x_opt.clear();
        for (auto& kv : all) {
            f_vals.push_back(kv.first);
            r_opt.push_back(get<0>(kv.second));
            n_opt.push_back(get<1>(kv.second));
            Fd_min.push_back(get<2>(kv.second));
            x_opt.push_back(get<3>(kv.second));
        }

        // 再次检查精度
        if (f_vals.size() > 1) {
            double min_diff = numeric_limits<double>::infinity();
            for (size_t i = 1; i < f_vals.size(); ++i)
                min_diff = min(min_diff, f_vals[i] - f_vals[i-1]);
            if (min_diff <= p.tolerance) break;
        }
    }
    return { f_vals, r_opt, n_opt, Fd_min, x_opt };
}

// ===================== ΔE 采样（截断正态分布） =====================
vector<double> generate_DeltaE(int n, const SimParams& p, mt19937& gen) {
    normal_distribution<double> dist(p.E_mean, p.E_std);
    vector<double> samples;
    while (samples.size() < static_cast<size_t>(n)) {
        double val = dist(gen);
        if (val >= p.lower_bound() && val <= p.upper_bound())
            samples.push_back(val);
    }
    return samples;
}

// ===================== 单条链全流程 =====================
ChainResult optimize_single_chain(int chain_idx, const SimParams& p, mt19937& gen) {
    vector<double> deltaE = generate_DeltaE(p.N_domains, p, gen);

    // 逐个domain自适应优化
    vector<DomainResult> domain_results;
    for (int i = 0; i < p.N_domains; ++i) {
        DomainResult dr;
        dr.domain_idx = i;
        dr.DeltaE = deltaE[i];
        dr.adaptive_result = optimize_single_domain_adaptive(deltaE[i], p);
        domain_results.push_back(dr);
    }

    // 构建统一力网格（并集）
    set<double> all_f_set;
    for (auto& dr : domain_results)
        for (double f : dr.adaptive_result.f_vals)
            all_f_set.insert(f);
    vector<double> unified_f(all_f_set.begin(), all_f_set.end());
    sort(unified_f.begin(), unified_f.end());

    // 在统一网格上重新计算每个domain
    int n_f = unified_f.size();
    vector<vector<double>> unified_r(n_f, vector<double>(p.N_domains));
    vector<vector<double>> unified_n(n_f, vector<double>(p.N_domains));
    vector<vector<double>> unified_F(n_f, vector<double>(p.N_domains));
    vector<vector<double>> unified_x(n_f, vector<double>(p.N_domains));

    for (int i = 0; i < p.N_domains; ++i) {
        for (int j = 0; j < n_f; ++j) {
            auto res = optimize_single_point(unified_f[j], deltaE[i], p);
            unified_r[j][i] = res.r_best;
            unified_n[j][i] = res.n_best;
            unified_F[j][i] = res.Fd_min;
            unified_x[j][i] = res.x_best;
        }
    }

    // 转置为按domain存储 (N_domains x n_f)
    vector<vector<double>> unified_r_opt(p.N_domains, vector<double>(n_f));
    vector<vector<double>> unified_n_opt(p.N_domains, vector<double>(n_f));
    vector<vector<double>> unified_Fd_min(p.N_domains, vector<double>(n_f));
    vector<vector<double>> unified_x_opt(p.N_domains, vector<double>(n_f));
    for (int i = 0; i < p.N_domains; ++i)
        for (int j = 0; j < n_f; ++j) {
            unified_r_opt[i][j] = unified_r[j][i];
            unified_n_opt[i][j] = unified_n[j][i];
            unified_Fd_min[i][j] = unified_F[j][i];
            unified_x_opt[i][j] = unified_x[j][i];
        }

    ChainResult cr;
    cr.chain_idx = chain_idx;
    cr.DeltaE = deltaE;
    cr.unified_f_vals = unified_f;
    cr.unified_r_opt = unified_r_opt;
    cr.unified_n_opt = unified_n_opt;
    cr.unified_Fd_min = unified_Fd_min;
    cr.unified_x_opt = unified_x_opt;
    cr.domain_results = domain_results;
    return cr;
}

// ===================== 文件保存函数 =====================
void save_matrix_csv(const vector<vector<double>>& data,
                     const vector<double>& f_vals,
                     const string& filename, int n_domains)
{
    ofstream file(filename);
    file << scientific << setprecision(15);
    file << ",";
    for (int i = 1; i <= n_domains; ++i) {
        file << "Domain_" << i;
        if (i < n_domains) file << ",";
    }
    file << "\n";
    for (size_t j = 0; j < f_vals.size(); ++j) {
        file << f_vals[j];
        for (int i = 0; i < n_domains; ++i)
            file << "," << data[i][j];
        file << "\n";
    }
}

void save_chain_results(const ChainResult& chain, const string& save_dir, const SimParams& p) {
    string prefix = save_dir + "/chain_" + to_string(chain.chain_idx + 1);
    save_matrix_csv(chain.unified_r_opt, chain.unified_f_vals, prefix + "_r_values_unified.csv", p.N_domains);
    save_matrix_csv(chain.unified_n_opt, chain.unified_f_vals, prefix + "_n_values_unified.csv", p.N_domains);
    save_matrix_csv(chain.unified_Fd_min, chain.unified_f_vals, prefix + "_Fd_values_unified.csv", p.N_domains);
    save_matrix_csv(chain.unified_x_opt, chain.unified_f_vals, prefix + "_x_values_unified.csv", p.N_domains);
}

void save_all_DeltaE(const vector<vector<double>>& all_DeltaE, const string& filename) {
    ofstream file(filename);
    file << scientific << setprecision(15);
    int n_domains = all_DeltaE.empty() ? 0 : all_DeltaE[0].size();
    file << ",";
    for (int j = 1; j <= n_domains; ++j) {
        file << "Domain_" << j;
        if (j < n_domains) file << ",";
    }
    file << "\n";
    for (size_t i = 0; i < all_DeltaE.size(); ++i) {
        file << "Chain_" << (i+1);
        for (double e : all_DeltaE[i]) file << "," << e;
        file << "\n";
    }
}

void save_params(const SimParams& p, const string& filename) {
    ofstream file(filename);
    file << "Parameter,Value,Description\n";
    file << "xi_f," << p.xi_f << ",折叠态长度\n";
    file << "alpha," << p.alpha << ",解折叠长度系数\n";
    file << "E_mean," << p.E_mean << ",解折叠能均值\n";
    file << "E_std," << p.E_std << ",解折叠能标准差\n";
    file << "E_delta," << p.E_delta << ",能量截断半宽\n";
    file << "N_domains_per_chain," << p.N_domains << ",每条链domain数量\n";
    file << "N_chains," << p.M_chains << ",链数量\n";
    file << "n_grid," << p.n_grid << ",n方向网格数\n";
    file << "f_grid_initial," << p.f_grid_initial << ",初始力网格数\n";
    file << "f_max," << p.f_max << ",最大力值\n";
    file << "refinement_threshold," << p.refinement_threshold << ",力细化阈值\n";
    file << "max_refinement_level," << p.max_refinement_level << ",最大细化层数\n";
    file << "tolerance," << p.tolerance << ",最小力间隔\n";
    file << "upper_bound," << p.upper_bound() << ",ΔE上界\n";
    file << "lower_bound," << p.lower_bound() << ",ΔE下界\n";
}

void save_statistics(const vector<ChainResult>& all_chains,
                     const vector<vector<double>>& all_DeltaE,
                     const string& filename, const SimParams& p)
{
    int M = all_chains.size();
    int N = p.N_domains;
    vector<int> chain_points;
    for (auto& ch : all_chains) chain_points.push_back(ch.unified_f_vals.size());

    double sum_pts = 0, min_pts = *min_element(chain_points.begin(), chain_points.end());
    double max_pts = *max_element(chain_points.begin(), chain_points.end());
    for (int pts : chain_points) sum_pts += pts;
    double mean_pts = sum_pts / M;
    double sq_sum = 0;
    for (int pts : chain_points) sq_sum += (pts - mean_pts) * (pts - mean_pts);
    double std_pts = sqrt(sq_sum / M);

    vector<double> all_flat;
    for (auto& chain : all_DeltaE) for (double e : chain) all_flat.push_back(e);
    double sum_E = 0, min_E = all_flat[0], max_E = all_flat[0];
    for (double e : all_flat) {
        sum_E += e;
        if (e < min_E) min_E = e;
        if (e > max_E) max_E = e;
    }
    double mean_E = sum_E / all_flat.size();
    double sq_E = 0;
    for (double e : all_flat) sq_E += (e - mean_E) * (e - mean_E);
    double std_E = sqrt(sq_E / all_flat.size());

    ofstream file(filename);
    file << "Statistic,Value\n";
    file << "总链数," << M << "\n";
    file << "每条链domain数," << N << "\n";
    file << "平均力值点数," << mean_pts << "\n";
    file << "最小力值点数," << min_pts << "\n";
    file << "最大力值点数," << max_pts << "\n";
    file << "力值点数标准差," << std_pts << "\n";
    file << "平均ΔE," << mean_E << "\n";
    file << "ΔE标准差," << std_E << "\n";
    file << "最小ΔE," << min_E << "\n";
    file << "最大ΔE," << max_E << "\n";
}

// ===================== 主函数（并行执行） =====================
int main() {
    SimParams params;
    // 可根据需要修改参数
    // params.M_chains = 10;  // 测试时可降低

    string save_path = "./N_" + to_string(params.N_domains) + "_M_" + to_string(params.M_chains) + "_test_results";
    fs::create_directories(save_path);

    cout << "==============================================================\n";
    cout << "开始模拟 " << params.M_chains << " 条链，每条链 " << params.N_domains << " 个domain\n";
    cout << "ξ_f: " << params.xi_f << "\n";
    cout << "ΔE分布: 均值=" << params.E_mean << ", 标准差=" << params.E_std
         << ", 截断=[" << params.lower_bound() << ", " << params.upper_bound() << "]\n";
    cout << "力值范围: [0, " << params.f_max << "]\n";
    cout << "保存路径: " << save_path << "\n";
    cout << "使用 OpenMP 并行，线程数: " << omp_get_max_threads() << "\n";
    cout << "==============================================================\n";

    // 为每条链准备独立的随机数生成器（种子不同）
    vector<mt19937> generators(params.M_chains);
    random_device rd;
    for (int i = 0; i < params.M_chains; ++i) {
        generators[i].seed(rd());
    }

    vector<ChainResult> all_chain_results(params.M_chains);
    vector<vector<double>> all_DeltaE(params.M_chains);

    // ====== OpenMP 并行循环 ======
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < params.M_chains; ++i) {
        // 每个线程独立输出进度（加锁避免混乱）
        #pragma omp critical
        {
            cout << "进度: " << (i+1) << "/" << params.M_chains
                 << " (线程 " << omp_get_thread_num() << ")\n";
        }

        ChainResult cr = optimize_single_chain(i, params, generators[i]);
        all_chain_results[i] = cr;
        all_DeltaE[i] = cr.DeltaE;
        save_chain_results(cr, save_path, params);
    }

    // 保存汇总文件
    cout << "\n保存所有链的ΔE参数...\n";
    save_all_DeltaE(all_DeltaE, save_path + "/all_chains_DeltaE.csv");
    save_params(params, save_path + "/simulation_parameters.csv");
    save_statistics(all_chain_results, all_DeltaE, save_path + "/simulation_statistics.csv", params);

    cout << "\n模拟完成！结果保存在 '" << save_path << "'\n";
    return 0;
}