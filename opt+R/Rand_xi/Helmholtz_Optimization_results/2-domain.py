"""
目的：探究不同参数下的domain的解折叠顺序，参数包括xi_f1,xi_f2,E1,E2
系统：包含2个domain
主题：约束下优化系统的Helmholtz自由能
新增：可视化部分，绘制n1,n2和力曲线，风格与参考代码一致
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import minimize_scalar

# ============ 字体路径（如有需要可修改） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 45
label_fontsize = 45
tick_fontsize = 45
legend_fontsize = 25
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 8
ytick_major_width = 8
xtick_major_size = 8
ytick_major_size = 8
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 5
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

# ============ 物理参数 ============
xi_f1 = 5.0          # 第一个domain的折叠态长度
alpha = 7.0              # alpha = xi_ui/xi_fi
beta = 1.5           # beta = xi_f2/xi_f1
force_limit = 250.0    # 力曲线y轴上限
E0 = 220.0
delta = 20.0

# 优化参数
r_grids = 1000

# 设置存储路径
save_path = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_results"
os.makedirs(save_path, exist_ok=True)

# ---------- 辅助函数（优化部分） ----------
def energy_term_U(n_i, DeltaEi):
    """
    能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)
    """
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """
    轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (alpha - 1)ξ_fi
    """
    return xi_fi + n_i * (alpha - 1) * xi_fi

def end_to_end_factor_x_i(r_i, n_i, xi_fi):
    """
    端到端因子: x_i(r_i, n_i) = r_i / L_{ci}(n_i)
    """
    L_ci = contour_length_Lci(n_i, xi_fi)
    return r_i / L_ci

def WLC_free_energy(x_i, L_ci):
    if x_i >= 1.0 - 1e-12:          # 更保守的阈值
        return 1e300                 # 极大有限值
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, xi_fi):
    L_ci = contour_length_Lci(n_i, xi_fi)
    if r_i < 0 or r_i > L_ci:
        return 1e300
    x_i = r_i / L_ci
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    return F_wlc + Ui

def free_energy_2_domain(r, r1, n1, n2):
    # 第1个domain的能量
    delta_E1 = E0
    energy1 = single_domain_free_energy(r1, n1, delta_E1, xi_f1)
    # 第2个domain的能量
    r2 = r - r1
    xi_f2 = beta * xi_f1
    delta_E2 = E0 + delta
    energy2 = single_domain_free_energy(r2, n2, delta_E2, xi_f2)
    total = energy1 + energy2
    if not np.isfinite(total):
        return 1e300
    return total

def Optimize_single_point(r, init_points=15, refine_levels=6, refine_points=15, tol=1e-6):
    """
    使用粗网格 + 局部细化优化给定拉伸 r 下的自由能。
    返回 [r, r1_opt, n1_opt, n2_opt]
    """
    n1_min, n1_max = 0.0, 1.0
    n2_min, n2_max = 0.0, 1.0
    best_r1 = None
    best_n1 = None
    best_n2 = None
    best_F = float('inf')

    for level in range(refine_levels + 1):
        # 当前层网格点数
        N = init_points if level == 0 else refine_points
        n1_grid = np.linspace(n1_min, n1_max, N)
        n2_grid = np.linspace(n2_min, n2_max, N)

        level_best_F = float('inf')
        level_best_r1 = None
        level_best_n1 = None
        level_best_n2 = None

        # 遍历当前网格
        for n1 in n1_grid:
            for n2 in n2_grid:
                Lc1 = contour_length_Lci(n1, xi_f1)
                Lc2 = contour_length_Lci(n2, beta * xi_f1)
                r1_lower = max(0.0, r - min(r, Lc2))
                r1_upper = min(Lc1, r)
                if r1_lower > r1_upper:
                    continue  # 不可行

                # 对 r1 进行一维凸优化
                res = minimize_scalar(
                    lambda r1: free_energy_2_domain(r, r1, n1, n2),
                    bounds=(r1_lower, r1_upper),
                    method='bounded',
                    options={'xatol': 1e-8, 'maxiter': 100}
                )
                if res.success and res.fun < level_best_F:
                    level_best_F = res.fun
                    level_best_r1 = res.x
                    level_best_n1 = n1
                    level_best_n2 = n2

        # 如果当前层无可行点，直接返回 NaN
        if level_best_F == float('inf'):
            return np.array([r, np.nan, np.nan, np.nan])

        # 更新全局最优
        if level_best_F < best_F:
            best_F = level_best_F
            best_r1 = level_best_r1
            best_n1 = level_best_n1
            best_n2 = level_best_n2

        # 最后一级细化结束
        if level == refine_levels:
            break

        # 确定下一级细化的范围：以当前最优为中心，取相邻网格点间距
        idx_n1 = np.argmin(np.abs(n1_grid - best_n1))
        idx_n2 = np.argmin(np.abs(n2_grid - best_n2))

        # 获取左右邻居（边界处理）
        left_n1 = n1_grid[max(0, idx_n1-1)]
        right_n1 = n1_grid[min(N-1, idx_n1+1)]
        left_n2 = n2_grid[max(0, idx_n2-1)]
        right_n2 = n2_grid[min(N-1, idx_n2+1)]

        n1_min, n1_max = left_n1, right_n1
        n2_min, n2_max = left_n2, right_n2

        # 检查是否达到精度
        if (n1_max - n1_min < tol) and (n2_max - n2_min < tol):
            break

    return np.array([r, best_r1, best_n1, best_n2])

# ---------- 可视化函数（新增） ----------
def MS_force(r, L_c):
    """
    Marko–Siggia 力（WLC模型近似）
    """
    x = np.asarray(r, dtype=float) / np.asarray(L_c, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     1e15)
    return force

def plot_n_curves(ax, r, n1, n2, title):
    """绘制n1, n2 vs r曲线"""
    ax.plot(r, n1, color='blue', linewidth=lines_linewidth, label='$n_1$', zorder=3)
    ax.plot(r, n2, color='red', linewidth=lines_linewidth, linestyle='--', label='$n_2$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$n$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    # 刻度设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

def plot_force_curves(ax, r, force1, force2, title):
    """绘制f1, f2 vs r曲线"""
    ax.plot(r, 5.1*force1, color='blue', linewidth=lines_linewidth, label='$f_1$', zorder=3)
    ax.plot(r, 5.1*force2, color='black', linewidth=lines_linewidth, linestyle='-', label='$f_2$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$f \quad [\\text{pN}]$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    # 刻度设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    ax.set_xlim(0, r[-1])
    ax.set_ylim(0, force_limit)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

# ---------- 主程序 ----------
def main():
    # 计算拉伸的最大可能值（两个域完全伸展时的总轮廓长度）
    r_max = contour_length_Lci(1, xi_f1) + contour_length_Lci(1, beta * xi_f1)
    r_vals = np.linspace(0, 0.95 * r_max, r_grids)

    results = []
    for i, r in enumerate(r_vals):
        opt_result = Optimize_single_point(r)
        r_val, r1_opt, n1_opt, n2_opt = opt_result
        r2_opt = r_val - r1_opt if not np.isnan(r1_opt) else np.nan
        results.append([r_val, r1_opt, r2_opt, n1_opt, n2_opt])

        # 输出进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1} / {len(r_vals)} 个点")

    # 保存为 CSV 文件
    df = pd.DataFrame(results, columns=['r', 'r1', 'r2', 'n1', 'n2'])
    csv_filename = os.path.join(save_path, "2_domain_results.csv")
    df.to_csv(csv_filename, index=False)
    print(f"结果已保存至: {csv_filename}")

    # ===== 可视化 =====
    # 提取数据
    r = df['r'].values
    r1 = df['r1'].values
    r2 = df['r2'].values
    n1 = df['n1'].values
    n2 = df['n2'].values

    # 计算轮廓长度和力
    xi_f2 = beta * xi_f1
    Lc1 = contour_length_Lci(n1, xi_f1)
    Lc2 = contour_length_Lci(n2, xi_f2)
    force1 = MS_force(r1, Lc1)
    force2 = MS_force(r2, Lc2)

    # 将无穷大力替换为NaN
    force1 = np.where(np.isfinite(force1), force1, np.nan)
    force2 = np.where(np.isfinite(force2), force2, np.nan)

    # 标题
    #title = f"α = {alpha}, β = {beta}"
    title = f"$\Delta E_1 = {E0}, \Delta E_2 = {E0 + delta}$, β = 1.5"

    # 创建Figure文件夹
    output_dir = os.path.join(save_path, "Figure")
    os.makedirs(output_dir, exist_ok=True)

    # 绘制 n 曲线
    fig_n, ax_n = plt.subplots(1, 1, figsize=(12, 9))
    plot_n_curves(ax_n, r, n1, n2, title)
    plt.tight_layout()
    n_output = os.path.join(output_dir, "2_domain_results_n.png")
    plt.savefig(n_output, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig_n)
    print(f"n 曲线已保存至: {n_output}")

    # 绘制 force 曲线
    fig_f, ax_f = plt.subplots(1, 1, figsize=(12, 9))
    plot_force_curves(ax_f, r, force1, force2, title)
    plt.tight_layout()
    f_output = os.path.join(output_dir, "2_domain_results_force.png")
    plt.savefig(f_output, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig_f)
    print(f"force 曲线已保存至: {f_output}")

if __name__ == "__main__":
    main()