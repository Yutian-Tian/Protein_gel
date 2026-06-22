"""
系统：单个domain
目的：比较Gibbs和Helmholtz两种计算的力拉伸曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import minimize_scalar

# ============ 字体路径 ============
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

# ============ 统一物理参数 ============
xi_f = 5.0      # 折叠态轮廓长度
alpha = 7.0     # 解折叠后长度倍率
DeltaE = 13.0   # 能量差 ΔE
pi = np.pi

# 吉布斯优化（恒力系综）网格参数
f_min = 0.0
f_max = 6.0
f_grid = 200
r_grid = 400
n_grid_gibbs = 200

# 亥姆霍兹优化（恒距系综）精度参数
init_points = 25
refine_levels = 4
opt_tol = 1e-8
r_num = 500

# ============ 公共物理函数 ============
def Lc(n):
    """轮廓长度: Lc(n) = ξ_f + n·(α-1)·ξ_f"""
    return xi_f + n * (alpha - 1) * xi_f

def U(n):
    """周期性势能项: U(n) = ΔE·n - ΔE·cos(2πn)"""
    return DeltaE * n - DeltaE * np.cos(2 * pi * n)

def F_MS(r, n):
    """Marko-Siggia WLC自由能"""
    Lc_val = Lc(n)
    x_val = r / Lc_val
    x_clipped = np.clip(x_val, 0.001, 0.999)
    return 0.25 * Lc_val * x_clipped**2 * (3 - 2 * x_clipped) / (1 - x_clipped)

def force_MS(r, n):
    """Marko-Siggia 力-距关系"""
    Lc_val = Lc(n)
    x_val = r / Lc_val
    x_clipped = np.clip(x_val, 0.001, 0.999)
    return 0.25 * ((1 - x_clipped)**(-2) + 4 * x_clipped - 1.0)

def F_total(r, n):
    """总亥姆霍兹自由能"""
    return F_MS(r, n) + U(n)

def F_modified(r, n, f):
    """吉布斯自由能：F(r,n) - f·r"""
    return F_total(r, n) - f * r

# ============ 方法1：恒力系综（吉布斯自由能最小化） ============
def simulate_gibbs(f_values, r_points=200, n_points=100):
    """固定外力f，扫描r和n最小化吉布斯自由能 G = F - f·r"""
    n_sim = []
    r_sim = []
    
    r_min = 0.0
    r_max = 0.95 * alpha * xi_f
    r_vals = np.linspace(r_min, r_max, r_points)
    n_vals = np.linspace(0, 1.0, n_points)
    
    for f in f_values:
        min_F = np.inf
        r_opt = 0.0
        n_opt = 0.0
        
        for r in r_vals:
            for n in n_vals:
                F_mod = F_modified(r, n, f)
                if np.isfinite(F_mod) and F_mod < min_F:
                    min_F = F_mod
                    r_opt = r
                    n_opt = n
        
        r_sim.append(r_opt)
        n_sim.append(n_opt)
    
    return np.array(r_sim), np.array(n_sim)

# ============ 方法2：恒距系综（亥姆霍兹自由能最小化） ============
def optimize_n_for_r(r):
    """固定端距r，优化n最小化亥姆霍兹自由能 F(r,n)"""
    n_min, n_max = 0.0, 1.0
    best_n = None
    best_F = float('inf')

    for level in range(refine_levels + 1):
        n_grid = np.linspace(n_min, n_max, init_points)
        level_best_F = float('inf')
        level_best_n = None

        for n in n_grid:
            F_val = F_total(r, n)
            if F_val < level_best_F:
                level_best_F = F_val
                level_best_n = n

        if level_best_F == float('inf'):
            return np.nan

        idx = np.argmin(np.abs(n_grid - level_best_n))
        left = n_grid[max(0, idx - 1)]
        right = n_grid[min(init_points - 1, idx + 1)]

        res = minimize_scalar(
            lambda n: F_total(r, n),
            bounds=(left, right),
            method='bounded',
            options={'xatol': opt_tol, 'maxiter': 200}
        )

        if res.success and res.fun < best_F:
            best_F = res.fun
            best_n = res.x

        if level == refine_levels:
            break

        n_min, n_max = left, right
        if (n_max - n_min) < opt_tol:
            break

    return best_n

def simulate_helmholtz(r_values):
    """恒距系综全量计算"""
    n_sim = []
    f_sim = []

    for i, r in enumerate(r_values):
        n_opt = optimize_n_for_r(r)
        if np.isnan(n_opt):
            n_sim.append(np.nan)
            f_sim.append(np.nan)
        else:
            f_opt = force_MS(r, n_opt)
            n_sim.append(n_opt)
            f_sim.append(f_opt)

        if (i + 1) % 100 == 0:
            print(f"亥姆霍兹计算进度: {i+1} / {len(r_values)}")

    return np.array(n_sim), np.array(f_sim)

# ============ 主程序：计算+合并绘图 ============
def main():
    # 1. 吉布斯系综计算（恒力）
    print("===== 开始吉布斯系综计算（恒力） =====")
    f_values = np.linspace(f_min, f_max, f_grid)
    r_gibbs, n_gibbs = simulate_gibbs(f_values, r_points=r_grid, n_points=n_grid_gibbs)
    print("吉布斯系综计算完成\n")

    # 2. 亥姆霍兹系综计算（恒距）
    print("===== 开始亥姆霍兹系综计算（恒距） =====")
    r_min = 0.01
    r_max = 0.95 * alpha * xi_f
    r_values_helm = np.linspace(r_min, r_max, r_num)
    n_helm, f_helm = simulate_helmholtz(r_values_helm)
    print("亥姆霍兹系综计算完成\n")

    # 3. 合并绘制 f-r 曲线
    print("===== 绘制合并力拉伸曲线 =====")
    fig, ax = plt.subplots(figsize=(10, 8))

    # 吉布斯系综曲线
    ax.plot(r_gibbs, f_values, color='red', linewidth=lines_linewidth,
            label='Gibbs')
    
    # 亥姆霍兹系综曲线
    ax.plot(r_values_helm, f_helm, color='blue', linewidth=lines_linewidth,
            linestyle='--', label='Helmholtz')

    # 坐标轴与标题
    ax.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax.set_title('Force-extension: Gibbs vs. Helmholtz',
                 fontsize=title_fontsize, pad=20)

    # 网格与图例
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    # 坐标范围
    ax.set_xlim(0, alpha * xi_f)
    ax.set_ylim(-0.5, f_max)

    # 刻度设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()

    # 边框强化
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()

    # 保存图片
    save_path = '/home/tyt/project/Single-chain/Gibbs_vs_Helmholtz_fr'
    plt.savefig(f'{save_path}.png', dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"合并图已保存至: {save_path}.png")

    # 输出参数汇总
    print("\n" + "="*80)
    print("系统统一参数:")
    print(f"  ξ_f = {xi_f}")
    print(f"  α = {alpha}")
    print(f"  ΔE = {DeltaE}")
    print(f"\n吉布斯系综: 力范围 [{f_min}, {f_max}], 共 {f_grid} 个力点")
    print(f"亥姆霍兹系综: 距离范围 [{r_min:.2f}, {r_max:.2f}], 共 {r_num} 个距离点")
    print("="*80)


if __name__ == "__main__":
    main()