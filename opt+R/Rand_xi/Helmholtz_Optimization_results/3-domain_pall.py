"""
目的：探究不同参数下的三个domain的解折叠顺序（并行加速版，无tqdm）
系统：包含3个domain，使用多进程并行优化
优化策略：版本A的回退网格凸优化 + 三维网格细化
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import minimize, LinearConstraint
from concurrent.futures import ProcessPoolExecutor, as_completed

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
lines_linewidth = 1
lines_markersize = 15

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

# ============ 物理参数（版本B） ============
xi_f1 = 5.0          # 第一个domain的折叠态长度
k = 7.0              # k = xi_ui/xi_fi
beta2 = 1.5          # xi_f2 / xi_f1
beta3 = 2.0          # xi_f3 / xi_f1
force_limit = 20.0   # 力曲线y轴上限
E0 = 2.0
Ek = 5.0             # 能量系数

# 三个域的具体参数
xi_f2 = beta2 * xi_f1
xi_f3 = beta3 * xi_f1
DeltaE1 = E0
DeltaE2 = E0 + Ek*(xi_f2 - xi_f1)
DeltaE3 = E0 + Ek*(xi_f3 - xi_f1)

xi_f_list = [xi_f1, xi_f2, xi_f3]
DeltaE_list = [DeltaE1, DeltaE2, DeltaE3]

# 优化参数
r_grids = 1000

# 外层搜索参数（版本A风格）
init_points = 10      # 粗网格点数（每个维度），9^3=729个初始点
refine_levels = 8    # 细化层数
refine_points = 10   # 每层细化点数（每个维度）
tol = 1e-6           # 收敛容差

# 并行参数
n_workers = 8     # None表示使用所有CPU核心，也可指定数字如8

# 设置存储路径
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/3_domain_results_pall"
os.makedirs(save_path, exist_ok=True)


# ---------- 辅助函数（与串行版相同，必须在顶层以支持pickle） ----------
def contour_length_Lci(n_i, xi_fi):
    return xi_fi + n_i * (k - 1) * xi_fi


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


def total_free_energy_3domain(r, r1, r2, n1, n2, n3):
    r3 = r - r1 - r2
    F1 = single_domain_free_energy(r1, n1, DeltaE_list[0], xi_f_list[0])
    F2 = single_domain_free_energy(r2, n2, DeltaE_list[1], xi_f_list[1])
    F3 = single_domain_free_energy(r3, n3, DeltaE_list[2], xi_f_list[2])
    total = F1 + F2 + F3
    if not np.isfinite(total):
        return 1e300
    return total


def optimize_r1r2_given_n(r, n1, n2, n3):
    L1 = contour_length_Lci(n1, xi_f_list[0])
    L2 = contour_length_Lci(n2, xi_f_list[1])
    L3 = contour_length_Lci(n3, xi_f_list[2])

    if r < 0 or r > L1 + L2 + L3:
        return np.nan, np.nan, 1e300

    def objective(r12):
        r1v, r2v = r12[0], r12[1]
        return total_free_energy_3domain(r, r1v, r2v, n1, n2, n3)

    bounds = [(0, L1), (0, L2)]
    A = [[1, 1], [-1, -1]]
    b_ub = [r, -(r - L3)]
    linear_constraint = LinearConstraint(A, lb=-np.inf, ub=b_ub)

    r1_guess = min(L1, max(0, r / 3))
    r2_guess = min(L2, max(0, (r - r1_guess) / 2))
    x0 = [r1_guess, r2_guess]

    try:
        res = minimize(objective, x0, method='SLSQP',
                       bounds=bounds, constraints=linear_constraint,
                       options={'ftol': 1e-9, 'disp': False})
        if res.success and np.isfinite(res.fun):
            return res.x[0], res.x[1], res.fun
        else:
            # 回退网格
            best_F = 1e300
            best_r1, best_r2 = np.nan, np.nan
            for r1_cand in np.linspace(max(0, r - L2 - L3), min(L1, r), 20):
                r2_max = min(L2, r - r1_cand)
                r2_min = max(0, r - r1_cand - L3)
                if r2_min > r2_max:
                    continue
                for r2_cand in np.linspace(r2_min, r2_max, 20):
                    F = objective([r1_cand, r2_cand])
                    if F < best_F:
                        best_F = F
                        best_r1, best_r2 = r1_cand, r2_cand
            return best_r1, best_r2, best_F
    except Exception:
        return np.nan, np.nan, 1e300


def Optimize_single_point_3domain(r):
    """对单个r进行优化，返回结果列表（供并行调用）"""
    n_min = np.array([0.0, 0.0, 0.0])
    n_max = np.array([1.0, 1.0, 1.0])
    best_n = np.array([0.5, 0.5, 0.5])
    best_r1, best_r2 = np.nan, np.nan
    best_F = float('inf')

    N = init_points
    for level in range(refine_levels + 1):
        n1_vals = np.linspace(n_min[0], n_max[0], N)
        n2_vals = np.linspace(n_min[1], n_max[1], N)
        n3_vals = np.linspace(n_min[2], n_max[2], N)

        level_best_F = float('inf')
        level_best_n = None
        level_best_r1 = None
        level_best_r2 = None

        for n1 in n1_vals:
            for n2 in n2_vals:
                for n3 in n3_vals:
                    r1_opt, r2_opt, F_val = optimize_r1r2_given_n(r, n1, n2, n3)
                    if F_val < level_best_F:
                        level_best_F = F_val
                        level_best_r1 = r1_opt
                        level_best_r2 = r2_opt
                        level_best_n = (n1, n2, n3)

        if level_best_F == float('inf'):
            return [r, np.nan, np.nan, np.nan, np.nan, np.nan]

        if level_best_F < best_F:
            best_F = level_best_F
            best_r1, best_r2 = level_best_r1, level_best_r2
            best_n = level_best_n

        if level == refine_levels:
            break

        idx1 = np.argmin(np.abs(n1_vals - best_n[0]))
        idx2 = np.argmin(np.abs(n2_vals - best_n[1]))
        idx3 = np.argmin(np.abs(n3_vals - best_n[2]))

        left1 = n1_vals[max(0, idx1-1)]
        right1 = n1_vals[min(N-1, idx1+1)]
        left2 = n2_vals[max(0, idx2-1)]
        right2 = n2_vals[min(N-1, idx2+1)]
        left3 = n3_vals[max(0, idx3-1)]
        right3 = n3_vals[min(N-1, idx3+1)]

        n_min = np.array([left1, left2, left3])
        n_max = np.array([right1, right2, right3])

        if np.max(n_max - n_min) < tol:
            break

        N = refine_points

    return [r, best_r1, best_r2, best_n[0], best_n[1], best_n[2]]


# ---------- 力计算与可视化函数 ----------
def MS_force(r_i, L_ci):
    x = np.asarray(r_i, dtype=float) / np.asarray(L_ci, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     1e15)
    return force


def plot_n_curves(ax, r, n1, n2, n3, title):
    ax.plot(r, n1, color='blue', linewidth=lines_linewidth, label='$n_1$', zorder=3)
    ax.plot(r, n2, color='red', linewidth=lines_linewidth, linestyle='--', label='$n_2$', zorder=3)
    ax.plot(r, n3, color='green', linewidth=lines_linewidth, linestyle='-.', label='$n_3$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$n$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)


def plot_force_curves(ax, r, force1, force2, force3, title):
    ax.plot(r, force1, color='blue', linewidth=lines_linewidth, label='$f_1$', zorder=3)
    ax.plot(r, force2, color='red', linewidth=lines_linewidth, linestyle='--', label='$f_2$', zorder=3)
    ax.plot(r, force3, color='green', linewidth=lines_linewidth, linestyle='-.', label='$f_3$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$f$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    ax.set_xlim(0, r[-1])
    ax.set_ylim(0, force_limit)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)


# ---------- 主程序（并行部分，无tqdm） ----------
def main():
    # 最大可能拉伸
    r_max = (contour_length_Lci(1, xi_f1) +
             contour_length_Lci(1, xi_f2) +
             contour_length_Lci(1, xi_f3))
    r_vals = np.linspace(0, 0.95 * r_max, r_grids)

    # 并行计算所有r点的优化结果
    results = [None] * len(r_vals)  # 预分配保持顺序
    print("开始并行计算...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(Optimize_single_point_3domain, r): idx
                         for idx, r in enumerate(r_vals)}
        # 收集结果，每100个打印一次进度
        completed = 0
        total = len(r_vals)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error at r index {idx}: {e}")
                results[idx] = [r_vals[idx], np.nan, np.nan, np.nan, np.nan, np.nan]
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"已完成 {completed}/{total} 个点")

    # 将results整理为DataFrame
    data_rows = []
    for res in results:
        # res: [r, r1, r2, n1, n2, n3]
        r_val, r1_opt, r2_opt, n1_opt, n2_opt, n3_opt = res
        r3_opt = r_val - r1_opt - r2_opt if not np.isnan(r1_opt) else np.nan
        data_rows.append([r_val, r1_opt, r2_opt, r3_opt, n1_opt, n2_opt, n3_opt])

    df = pd.DataFrame(data_rows, columns=['r', 'r1', 'r2', 'r3', 'n1', 'n2', 'n3'])
    csv_filename = os.path.join(save_path, "3_domain_results.csv")
    df.to_csv(csv_filename, index=False)
    print(f"结果已保存至: {csv_filename}")

    # ===== 可视化（与串行版完全相同） =====
    r = df['r'].values
    n1 = df['n1'].values
    n2 = df['n2'].values
    n3 = df['n3'].values
    r1 = df['r1'].values
    r2 = df['r2'].values
    r3 = df['r3'].values

    Lc1 = contour_length_Lci(n1, xi_f1)
    Lc2 = contour_length_Lci(n2, xi_f2)
    Lc3 = contour_length_Lci(n3, xi_f3)
    force1 = MS_force(r1, Lc1)
    force2 = MS_force(r2, Lc2)
    force3 = MS_force(r3, Lc3)

    force1 = np.where(np.isfinite(force1), force1, np.nan)
    force2 = np.where(np.isfinite(force2), force2, np.nan)
    force3 = np.where(np.isfinite(force3), force3, np.nan)

    title = (f"$\\Delta E_1 = {DeltaE1:.1f},\\ \\Delta E_2 = {DeltaE2:.1f},\\ \\Delta E_3 = {DeltaE3:.1f}$\n"
             f"$\\beta_2 = {beta2},\\ \\beta_3 = {beta3}$")

    output_dir = os.path.join(save_path, "Figure")
    os.makedirs(output_dir, exist_ok=True)

    fig_n, ax_n = plt.subplots(1, 1, figsize=(12, 9))
    plot_n_curves(ax_n, r, n1, n2, n3, title)
    plt.tight_layout()
    n_output = os.path.join(output_dir, "3_domain_results_n.png")
    plt.savefig(n_output, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_n)
    print(f"n 曲线已保存至: {n_output}")

    fig_f, ax_f = plt.subplots(1, 1, figsize=(12, 9))
    plot_force_curves(ax_f, r, force1, force2, force3, title)
    plt.tight_layout()
    f_output = os.path.join(output_dir, "3_domain_results_force.png")
    plt.savefig(f_output, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig_f)
    print(f"force 曲线已保存至: {f_output}")


if __name__ == "__main__":
    main()