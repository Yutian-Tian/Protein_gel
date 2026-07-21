"""
绘制初始模量 G₀ 随 N 的变化，并与理论公式对比
R0与N存在关系 R0 = kR * N**0.5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ============ 字体设置 ============
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
k1 = 6.5
k2 = 1.50
kR = 0.5           # 初始首末端距离 R0 = kR * N**0.5

N_Area2_fixed = 8.0    # 部分解折叠模型中固定使用的 N
N_vals = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # 数值优化离散点
Rtheo_points = 200

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
    """应力优化计算"""
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
    """
    初态完全折叠时的理论 G₀ 解析解。
    严格遵循给定图片中的公式：
    G₀ = (3ρk_BT R₀ / 2l_p) * [ x₀/(2(1-x₀)³) + 1/(4(1-x₀)²) + 2x₀ - 1/4 ]
    其中 x₀ = R₀ / (N ξ_f)
    """
    L_c = N * xi_f
    x0 = R0 / L_c
    if x0 >= 1: 
        return 0
    
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
    xc = 0.541  
    f_prime_xc = MSforce_derivative(xc)
    A = (alpha - 1) / 2 * k1 * f_prime_xc
    B = (alpha + 1) / 2 - A * xc
    
    x0 = R0 / (N * xi_f)
    if x0 <= 0: return 0
    
    D = np.sqrt(max(0, B**2 + 4 * A * x0))
    x1 = (-B + D) / (2 * A)
    dx1 = x0 / D
    
    G0 = 1.5 * R0 * (MSforce(x1) + MSforce_derivative(x1) * dx1)
    return G0

# ===================== 可视化函数 =====================
def plot_G0_vs_N_area1(N_vals, save_dir=None):
    """
    绘制第一种情况 (完全折叠) G₀ 随 N 变化
    由于 R0 与 N 存在绑定关系（R0=kR*sqrt(N)），横轴改为 N
    """
    
    # 根据 N_vals 列表计算相应的 R0 列表
    R0_val = [kR * n**0.5 for n in N_vals]
    G0_val = [initial_modulus_num(r0, n) for r0, n in zip(R0_val, N_vals)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    # 数值优化解
    ax.plot(N_vals, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)
    
    # 理论解析解
    # 【核心修正】因为横坐标是 N，我们需要生成连续的 N_theo 作为 x 轴
    # 并通过 R0 = kR * sqrt(N) 求出对应的 R0 传入理论公式
    N_theo = np.linspace(0.5, 10.5, Rtheo_points)
    G0_theo = [calculate_G0_area1(kR * np.sqrt(N), N) for N in N_theo]
    
    ax.plot(N_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical', alpha=0.8, zorder=5)

    ax.set_xlabel('Number of domains $N$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Initial modulus $G_0$ vs. $N$ ($R_0 = {kR} \\sqrt{{N}}$)', fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'Initial1_G0_vs_N_kr={kR}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Area1 曲线已保存至: {path}")

def plot_G0_vs_R0_area2(R0_val, save_dir=None):
    """（备用）绘制第二种情况 G₀ 随 R₀ 变化"""
    G0_val = [initial_modulus_num(r0, N_Area2_fixed) for r0 in R0_val]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(N_vals, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)
    
    R0_theo = np.linspace(0.1, float(N_vals[-1]), Rtheo_points)
    G0_theo = [calculate_G0_area2(r0, N_Area2_fixed) for r0 in R0_theo]
    ax.plot(R0_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical (Area 2)', alpha=0.8, zorder=5)
    
    start_R0 = float(R0_theo[0])
    end_R0 = float(R0_theo[-1])
    end_G0 = float(G0_theo[-1])
    ref_x = [start_R0, end_R0]
    ref_y = [end_G0 * (start_R0 / end_R0)**2, end_G0]
    ax.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5, label=r'$G_0 \propto R_0^2$', zorder=3)

    ax.set_xlabel('Initial end-to-end distance $R_0$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Area 2: Partially Unfolded ($N={int(N_Area2_fixed)}$)', fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.set_xlim(1.0, float(N_vals[-1]) + 1.0)
    ax.set_ylim(0.1, 100.0)
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'Initial2_G0_vs_R0_N={int(N_Area2_fixed)}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Area2 曲线已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_N0.5'
    
    # 绘制基于 R0 = kR*sqrt(N) 的标度律对比 (以 N 为横坐标)
    plot_G0_vs_N_area1(N_vals, save_dir=output_dir)

if __name__ == "__main__":
    main()