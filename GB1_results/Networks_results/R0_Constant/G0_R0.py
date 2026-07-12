"""
绘制初始模量 G₀ 随初始末端距 R₀ 的变化，并与理论公式对比
包含两种初态分析：
1. 完全折叠状态 (Fully folded)
2. 部分解折叠状态 (Partially unfolded)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ============ 字体设置（与之前脚本保持一致） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 30
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 4
lines_markersize = 15

figure_dpi = 100
savefig_dpi = 300
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

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

# ============ 核心物理参数 ============
xi_f = 3.6
alpha = 7.6
N_Area1 = 10.0
N_Area2 = 4.0
k1 = 6.5
k2 = 1.50

R0_val = [1.5, 5.0, 10.0, 15.0, 20.0, 25.0]   # 需要计算的 R₀ 点
Rtheo_points = 200                      # 理论曲线采样点数

# ===================== 基础物理函数 =====================
def Lc(f, N):
    return N * xi_f * (0.5 * (alpha + 1) + 0.5 * (alpha - 1) * np.tanh(k1 * (f - k2)))

def MSforce(x):
    return np.where(x < 0.99,
                    0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                    np.inf)

def MSforce_derivative(x):
    """Marko-Siggia 力关于 x 的导数 f'(x)"""
    return np.where(x < 0.99,
                    0.5 * (1 - x) ** (-3) + 1,
                    np.inf)

def StressOptimization(R0, N, r_val, f_val):
    Re = R0
    r_val = np.asarray(r_val)
    f_val = np.asarray(f_val)
    mask = r_val >= Re
    r_selected = r_val[mask]
    lambda_ = r_selected / Re
    r2 = lambda_ ** (-0.5) * Re
    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])
    sigma = Re * (f1 - lambda_ ** (-1.5) * f2)
    if np.abs(lambda_[0] - 1.0) > 1e-12:
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        sigma[0] = 0.0
    return lambda_, sigma

def initial_modulus_num(R0, N, fit_points=5):
    """数值计算初始模量 G₀"""
    x_MS = np.linspace(0.01, 0.99, 3000)
    f_MS = MSforce(x_MS)
    r_MS = x_MS * Lc(f_MS, N)
    lambda_, sigma = StressOptimization(R0, N, r_MS, f_MS)
    lam_fit = lambda_[:fit_points]
    sig_fit = sigma[:fit_points]
    A = (lam_fit - 1.0).reshape(-1, 1)
    G0, _, _, _ = np.linalg.lstsq(A, sig_fit, rcond=None)
    return G0[0]

# ===================== 第一种情况 (完全折叠) =====================
def calculate_G0_area1(R0, N):
    """初态完全折叠时的理论 G₀ 解析解"""
    L_c = N * xi_f
    x0 = R0 / L_c
    if x0 >= 1: return 0
    term1 = x0 / (2 * (1 - x0)**3)
    term2 = 1 / (4 * (1 - x0)**2)
    term3 = 2 * x0
    bracket = term1 + term2 + term3 - 0.25
    coeff = 1.5 * R0
    return coeff * bracket

# ===================== 第二种情况 (部分解折叠) =====================
def calculate_G0_area2(R0, N):
    """
    初态部分解折叠时的理论 G₀ 解析解。
    基于非线性轮廓长度模型： Ax^2 + Bx = x0
    """
    # 1. 求临界点 xc 使得 f(xc) = k2
    # 数值解方程 0.25 * ((1-x)^-2 - 1 + 4x) = 1.48
    xc = 0.541  
    f_prime_xc = MSforce_derivative(xc)
    
    # 2. 计算二次方程的系数 A 和 B
    A = (alpha - 1) / 2 * k1 * f_prime_xc
    B = (alpha + 1) / 2 - A * xc
    
    # 3. 确定初始比例 x0
    x0 = R0 / (N * xi_f)
    if x0 <= 0: return 0
    
    # 4. 解二次方程得到 λ=1 时的平衡伸长 x1
    D = np.sqrt(max(0, B**2 + 4 * A * x0))
    x1 = (-B + D) / (2 * A)
    
    # 5. 计算一阶导数 x1' = dx/dλ |_{λ=1}
    dx1 = x0 / D
    
    # 6. 代入 G0 解析解公式
    G0 = 1.5 * R0 * (MSforce(x1) + MSforce_derivative(x1) * dx1)
    return G0

# ===================== 可视化函数 =====================
def plot_G0_vs_R0_area1(R0_val, save_dir=None):
    """绘制第一种情况 (完全折叠) G₀ 随 R₀ 变化"""
    G0_val = [initial_modulus_num(r0, N_Area1) for r0 in R0_val]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 数值优化解
    ax.plot(R0_val, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)
    
    # 理论解析解
    R0_theo = np.linspace(0.1, float(R0_val[-1]), Rtheo_points)
    G0_theo = [calculate_G0_area1(r0, N_Area1) for r0 in R0_theo]
    ax.plot(R0_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical', alpha=0.8, zorder=5)
    
    # 标度参考线 G0 ∝ R0²
    ref_x = [R0_theo[0], R0_theo[-1]]
    ref_y = [G0_theo[0], G0_theo[0] * (R0_theo[-1] / R0_theo[0])**2]
    ax.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5, label=r'$G_0 \propto R_0^2$', zorder=3)

    ax.set_xlabel('Initial end-to-end distance $R_0$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Area 1: Fully Folded ($N={int(N_Area1)}$)', fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.set_xlim(1.0, float(R0_val[-1]) + 1.0)
    ax.set_ylim(0.1, 200.0)
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'Initial1_G0_vs_R0_N={int(N_Area1)}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Area1 曲线已保存至: {path}")

def plot_G0_vs_R0_area2(R0_val, save_dir=None):
    """绘制第二种情况 (部分解折叠) G₀ 随 R₀ 变化"""
    G0_val = [initial_modulus_num(r0, N_Area2) for r0 in R0_val]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 数值优化解
    ax.plot(R0_val, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)
    
    # 理论解析解
    R0_theo = np.linspace(0.1, float(R0_val[-1]), Rtheo_points)
    G0_theo = [calculate_G0_area2(r0, N_Area2) for r0 in R0_theo]
    ax.plot(R0_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical', alpha=0.8, zorder=5)
    
    # 【修改点】：标度参考线 G0 ∝ R0²，改为与右端点对齐
    start_R0 = float(R0_theo[0])
    end_R0 = float(R0_theo[-1])
    end_G0 = float(G0_theo[-1])
    ref_x = [start_R0, end_R0]
    ref_y = [end_G0 * (start_R0 / end_R0)**2, end_G0]
    ax.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5, label=r'$G_0 \propto R_0^2$', zorder=3)

    ax.set_xlabel('Initial end-to-end distance $R_0$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Initial modulus $G_0$ vs $R_0$ ($N={int(N_Area2)}$)', fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.set_xlim(1.0, float(R0_val[-1]) + 1.0)
    ax.set_ylim(0.1, 100.0)

    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'Initial2_G0_vs_R0_N={int(N_Area2)}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Area2 曲线已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_Constant'
    plot_G0_vs_R0_area1(R0_val, save_dir=output_dir)
    plot_G0_vs_R0_area2(R0_val, save_dir=output_dir)

if __name__ == "__main__":
    main()