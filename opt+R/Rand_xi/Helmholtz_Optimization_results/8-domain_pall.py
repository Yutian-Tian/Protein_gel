"""
目的：8个domain的解折叠顺序计算（并行加速版）
系统：包含8个domain，使用多进程并行优化
优化策略：随机采样初始点 + 局部凸优化（L-BFGS-B）
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import minimize, Bounds
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ============ 字体路径（如有需要可修改） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 25
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 5
lines_markersize = 5

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

figure_dpi = 100
savefig_dpi = 300

# ============ 应用全局设置 ============
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

plt.rcParams.update({
    'font.family': font_family,
    'mathtext.fontset': math_fontset,
    'mathtext.rm': math_rm,
    'mathtext.it': math_it,
    'mathtext.bf': math_bf,
    'font.weight': font_weight,
    'axes.titlesize': title_fontsize,
    'axes.labelsize': label_fontsize,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': legend_fontsize,
    'legend.title_fontsize': legend_title_fontsize,
    'axes.linewidth': axes_linewidth,
    'xtick.major.width': xtick_major_width,
    'ytick.major.width': ytick_major_width,
    'xtick.major.size': xtick_major_size,
    'ytick.major.size': ytick_major_size,
    'grid.linewidth': grid_linewidth,
    'grid.alpha': grid_alpha,
    'lines.linewidth': lines_linewidth,
    'lines.markersize': lines_markersize,
    'figure.dpi': figure_dpi,
    'savefig.dpi': savefig_dpi,
    'xtick.direction': xtick_direction,
    'ytick.direction': ytick_direction,
    'xtick.top': xtick_top,
    'ytick.right': ytick_right,
})

# ============ 物理参数（8个domain） ============
xi_f1 = 15.0          # 第一个domain的折叠态长度
alpha = 2.0           # alpha = xi_ui/xi_fi
# 定义 beta2...beta8
beta_values = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]   # 共7个，对应域2~8
force_limit = 8.0    # 力曲线y轴上限
E0 = 5.0
Ek = 2.0              # 能量系数

# 计算每个域的 xi_f 和 DeltaE
xi_f_list = [xi_f1]
DeltaE_list = [E0]
for i, beta in enumerate(beta_values, start=2):
    xi_fi = beta * xi_f1
    DeltaEi = E0 + Ek * (xi_fi - xi_f1)
    xi_f_list.append(xi_fi)
    DeltaE_list.append(DeltaEi)

n_domains = 8
assert len(xi_f_list) == n_domains and len(DeltaE_list) == n_domains

# 优化参数
r_grids = 500         # 力曲线上点的数量（可调）
max_iter_local = 200  # 局部优化最大迭代次数
tol_local = 1e-8      # 局部优化容差

# 并行参数
n_workers = 8

# 设置存储路径
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/8_domain_results_pall"
os.makedirs(save_path, exist_ok=True)


# ---------- 辅助函数（扩展到8个域） ----------
def contour_length_Lci(n_i, xi_fi):
    return xi_fi + n_i * (alpha - 1) * xi_fi

def energy_term_U(n_i, DeltaEi):
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def WLC_free_energy(x_i, L_ci):
    if x_i >= 1.0 - 1e-12:
        return 1e300
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i > L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def total_free_energy_8domain(r, r_vec, n_vec):
    """
    r_vec: 各域延伸长度 (r1,...,r8)
    n_vec: 各域折叠分数 (n1,...,n8)
    要求 sum(r_vec) = r
    """
    if abs(np.sum(r_vec) - r) > 1e-9:
        return 1e300
    total = 0.0
    for i in range(n_domains):
        total += single_domain_free_energy(r_vec[i], n_vec[i], DeltaE_list[i], xi_f_list[i])
        if not np.isfinite(total):
            return 1e300
    return total

def optimal_r_given_n(r, n_vec):
    """
    给定 n_vec，通过力平衡条件直接计算最优的 r_i 分配：
    各域拉伸比相等 => r_i / L_ci 相等 => r_i = r * L_ci / sum(L_ci)
    """
    Lc = np.array([contour_length_Lci(n_vec[i], xi_f_list[i]) for i in range(n_domains)])
    sum_Lc = np.sum(Lc)
    if sum_Lc <= 0:
        return np.full(n_domains, np.nan)
    r_opt = r * Lc / sum_Lc
    return r_opt

def objective_for_optimization(n_vec, r):
    """
    目标函数：给定n，先用最优分配计算r_i，再计算总自由能
    """
    r_opt = optimal_r_given_n(r, n_vec)
    if np.any(np.isnan(r_opt)):
        return 1e300
    # 确保所有 r_i <= 对应 L_ci
    Lc = np.array([contour_length_Lci(n_vec[i], xi_f_list[i]) for i in range(n_domains)])
    if np.any(r_opt > Lc + 1e-9):
        return 1e300
    return total_free_energy_8domain(r, r_opt, n_vec)

def Optimize_single_point_8domain(r, use_all_vertices=True, n_random_extra=0):
    """
    对单个r进行优化：
    1. 生成所有 0/1 顶点（256个）作为初始点
    2. 可选：额外生成 n_random_extra 个均匀随机点
    3. 每个初始点运行 L-BFGS-B 局部优化，取最优结果
    """
    bounds = [(0.0, 1.0) for _ in range(n_domains)]
    best_n = None
    best_F = float('inf')
    best_r_opt = None

    init_points = []

    # 1. 全部二进制顶点 (0/1 组合)
    if use_all_vertices:
        # 生成所有 2^n_domains 个组合
        for mask in range(1 << n_domains):
            n0 = np.array([(mask >> i) & 1 for i in range(n_domains)], dtype=float)
            init_points.append(n0)
    else:
        # 默认至少包含全0和全1
        init_points.append(np.zeros(n_domains))
        init_points.append(np.ones(n_domains))

    # 2. 额外的随机点
    np.random.seed()
    for _ in range(n_random_extra):
        init_points.append(np.random.uniform(0, 1, n_domains))

    # 去重（因为某些随机的可能和二进制重复，但概率极低，可省略）
    # 可选：打乱顺序，避免局部最优偏向某种模式
    # np.random.shuffle(init_points)

    for n0 in init_points:
        res = minimize(objective_for_optimization, n0, args=(r,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter_local, 'ftol': tol_local})
        if res.success and res.fun < best_F:
            best_F = res.fun
            best_n = res.x
            best_r_opt = optimal_r_given_n(r, best_n)

    if best_n is None:
        return [r] + [np.nan] * (2 * n_domains)

    return [r] + list(best_r_opt) + list(best_n)

# ---------- 力计算与可视化函数（8条曲线） ----------
def MS_force(r_i, L_ci):
    x = np.asarray(r_i, dtype=float) / np.asarray(L_ci, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     1e15)
    return force

def plot_n_curves(ax, r, n_matrix, title):
    """
    n_matrix: 形状 (n_r, n_domains) 的数组
    """
    colors = plt.cm.tab10(np.linspace(0, 1, n_domains))
    # 定义 8 种不同的标记样式（可根据需要增减）
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    for i in range(n_domains):
        ax.scatter(r, n_matrix[:, i], 
                   color=colors[i], marker=markers[i % len(markers)],
                   s=lines_markersize**2, label=f'$n_{i+1}$', 
                   zorder=3, alpha=0.7, edgecolors='none')
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$n$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best', ncol=2)
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

def plot_force_curves(ax, r, force_matrix, title):
    colors = plt.cm.tab10(np.linspace(0, 1, n_domains))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    for i in range(n_domains):
        ax.scatter(r, force_matrix[:, i],
                   color=colors[i], marker=markers[i % len(markers)],
                   s=lines_markersize**2, label=f'$f_{i+1}$',
                   zorder=3, alpha=0.7, edgecolors='none')
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$f$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best', ncol=2)
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    ax.set_xlim(0, r[-1])
    ax.set_ylim(0, force_limit)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

# ---------- 主程序 ----------
def main():
    # 最大可能拉伸
    r_max = sum(contour_length_Lci(1.0, xi_f) for xi_f in xi_f_list)
    r_vals = np.linspace(0, 0.95 * r_max, r_grids)

    print("开始并行计算 8-domain...")
    results = [None] * len(r_vals)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(Optimize_single_point_8domain, r): idx
                         for idx, r in enumerate(r_vals)}
        completed = 0
        total = len(r_vals)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error at r index {idx}: {e}")
                results[idx] = [r_vals[idx]] + [np.nan] * (2 * n_domains)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"已完成 {completed}/{total} 个点")

    # 整理数据
    columns = ['r'] + [f'r{i+1}' for i in range(n_domains)] + [f'n{i+1}' for i in range(n_domains)]
    data_rows = []
    for res in results:
        data_rows.append(res)
    df = pd.DataFrame(data_rows, columns=columns)
    csv_path = os.path.join(save_path, "8_domain_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"结果已保存至: {csv_path}")

    # 可视化
    r = df['r'].values
    n_arr = df[[f'n{i+1}' for i in range(n_domains)]].values
    r_arr = df[[f'r{i+1}' for i in range(n_domains)]].values

    # 计算力曲线
    force_arr = np.zeros_like(r_arr)
    for i in range(n_domains):
        Lc = contour_length_Lci(n_arr[:, i], xi_f_list[i])
        force_arr[:, i] = MS_force(r_arr[:, i], Lc)
        force_arr[:, i] = np.where(np.isfinite(force_arr[:, i]), force_arr[:, i], np.nan)

    # 标题：显示所有域的参数
    beta_str = ', '.join([f'$\\beta_{i+2}={beta_values[i]:.1f}$' for i in range(len(beta_values))])
    title = (f"8-domain results\n"
             f"$\\Delta E_1={DeltaE_list[0]:.1f}$, $\\Delta E_2={DeltaE_list[1]:.1f}$, ...\n"
             f"{beta_str}")

    output_dir = os.path.join(save_path, "Figure")
    os.makedirs(output_dir, exist_ok=True)

    fig_n, ax_n = plt.subplots(1, 1, figsize=(14, 9))
    plot_n_curves(ax_n, r, n_arr, title)
    plt.tight_layout()
    n_fig = os.path.join(output_dir, "8_domain_n.png")
    plt.savefig(n_fig, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig_n)
    print(f"n 曲线保存至: {n_fig}")

    fig_f, ax_f = plt.subplots(1, 1, figsize=(14, 9))
    plot_force_curves(ax_f, r, force_arr, title)
    plt.tight_layout()
    f_fig = os.path.join(output_dir, "8_domain_force.png")
    plt.savefig(f_fig, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig_f)
    print(f"力曲线保存至: {f_fig}")

if __name__ == "__main__":
    main()