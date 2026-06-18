'''
单个domain的模拟和理论对比 Delta理论
科研风格可视化
优化方法：粗网格全局搜索 + 多层细化 + 一维精确优化
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import minimize_scalar

# ============ 字体路径 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
# 字体设置
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

# 字体大小
title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 25
legend_title_fontsize = 35

# 线宽和尺寸
axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 5
lines_markersize = 15

# 刻度方向
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

# 图形设置
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
xi_f = 5.0      # 折叠态轮廓长度
alpha = 7.0     # 解折叠后长度倍率
DeltaE = 13.0   # 能量差 ΔE

# 优化精度参数
init_points = 25    # 初始粗网格点数
refine_levels = 4   # 局部细化层数
opt_tol = 1e-8      # 优化收敛精度

# ============ 核心物理函数（与图中公式严格对应） ============
def U(n):
    """周期性势能项: U(n) = ΔE·n - ΔE·cos(2πn)"""
    return DeltaE * n - DeltaE * np.cos(2 * np.pi * n)

def Lc(n):
    """轮廓长度: Lc(n) = ξ_f + n·(α-1)·ξ_f"""
    return xi_f + n * (alpha - 1) * xi_f

def WLC_free_energy(x, Lc_val):
    """
    WLC自由能（Marko-Siggia近似）
    对应图中公式: F_WLC = 0.25·Lc · x²(3-2x)/(1-x)
    """
    if x >= 1.0 - 1e-12:
        return 1e300  # 边界处返回有限大值，避免优化器异常
    return 0.25 * Lc_val * (x**2 * (3.0 - 2.0 * x)) / (1.0 - x)

def total_free_energy(n, r):
    """给定拉伸r下，总自由能作为n的函数（优化目标）"""
    Lc_val = Lc(n)
    if r < 0 or r > Lc_val:
        return 1e300
    x_val = r / Lc_val
    return WLC_free_energy(x_val, Lc_val) + U(n)

def MS_force(r, Lc_val):
    """
    Marko-Siggia 力: f = dF/dr
    公式: f = 0.25 · [ (1-x)⁻² - 1 + 4x ]
    """
    x = np.asarray(r, dtype=np.float64) / np.asarray(Lc_val, dtype=np.float64)
    force = np.where(x < 1.0,
                     0.25 * ((1.0 - x) ** (-2) - 1.0 + 4.0 * x),
                     1e15)
    return force

# ============ 高精度优化函数 ============
def Optimize_single_point(r):
    """
    对单个r值，优化n使总自由能最小
    策略：粗网格全局寻优 + 多层区间细化 + scipy精确一维优化
    """
    n_min, n_max = 0.0, 1.0
    best_n = None
    best_F = float('inf')

    for level in range(refine_levels + 1):
        # 当前层网格
        n_grid = np.linspace(n_min, n_max, init_points)
        level_best_F = float('inf')
        level_best_n = None

        # 遍历网格，找当前层最优
        for n in n_grid:
            F_val = total_free_energy(n, r)
            if F_val < level_best_F:
                level_best_F = F_val
                level_best_n = n

        # 无可行点
        if level_best_F == float('inf'):
            return np.nan

        # 在当前最优的邻域内做精确一维优化
        idx = np.argmin(np.abs(n_grid - level_best_n))
        left = n_grid[max(0, idx - 1)]
        right = n_grid[min(init_points - 1, idx + 1)]

        res = minimize_scalar(
            lambda n: total_free_energy(n, r),
            bounds=(left, right),
            method='bounded',
            options={'xatol': opt_tol, 'maxiter': 200}
        )

        if res.success and res.fun < best_F:
            best_F = res.fun
            best_n = res.x

        # 最后一层退出
        if level == refine_levels:
            break

        # 更新下一层细化范围
        n_min, n_max = left, right

        # 精度足够则提前收敛
        if (n_max - n_min) < opt_tol:
            break

    return best_n

def simulate_optimized(r_values):
    """遍历所有r值，完成高精度优化计算"""
    n_sim = []
    f_sim = []

    for i, r in enumerate(r_values):
        n_opt = Optimize_single_point(r)
        if np.isnan(n_opt):
            n_sim.append(np.nan)
            f_sim.append(np.nan)
        else:
            Lc_opt = Lc(n_opt)
            f_opt = MS_force(r, Lc_opt)
            n_sim.append(n_opt)
            f_sim.append(f_opt)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1} / {len(r_values)} 个点")

    return np.array(n_sim), np.array(f_sim)

# ============ 主程序 ============
def main():
    # 创建r值数组
    r_min = 0.01
    r_max = 0.95 * alpha * xi_f
    r_values = np.linspace(r_min, r_max, 500)

    # 高精度优化计算
    print("开始高精度优化计算...")
    n_sim, f_sim = simulate_optimized(r_values)
    print("计算完成")

    # ============ 科研风格可视化 ============
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 1. n-r 曲线
    ax1 = axes[0]
    ax1.plot(r_values, n_sim, color='red', linewidth=lines_linewidth, label='Optimization')

    ax1.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax1.set_ylabel('Unfolding probability $n$', fontsize=label_fontsize)
    ax1.set_title('$n$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    ax1.set_xlim(0, 19)
    ax1.set_ylim(-0.1, 1.1)

    ax1.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax1.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax1.minorticks_on()

    # 2. f-r 曲线
    ax2 = axes[1]
    ax2.plot(r_values, f_sim, color='red', linewidth=lines_linewidth, label='Optimization')

    ax2.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax2.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax2.set_title('Force $f$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    ax2.set_xlim(0, 35)
    ax2.set_ylim(-0.5, 6)

    ax2.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax2.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax2.minorticks_on()

    # 边框强化
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)

    plt.subplots_adjust(wspace=0.25, top=0.9, bottom=0.1)
    plt.tight_layout()

    # 保存图形
    base_save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/theory_vs_numerical'
    plt.savefig(f'{base_save_path}.png', dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n图形已保存至: {base_save_path}.png")

    # 输出参数信息
    print("\n" + "="*80)
    print("参数设置:")
    print(f"  ξ_f = {xi_f}")
    print(f"  α = {alpha}")
    print(f"  ΔE = {DeltaE}")
    print(f"  r 范围: {r_min:.2f} 到 {r_max:.2f}")
    print(f"  优化点数: {len(r_values)}")
    print(f"  优化精度: {opt_tol}")
    print("="*80)

if __name__ == "__main__":
    main()