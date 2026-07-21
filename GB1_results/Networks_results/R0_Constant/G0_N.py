"""
代码用于分析初始模量随 domain 数量 (N) 的变化
初态：(1) fully folded, (2) partially unfolded, (3) fully unfolded
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
import sys
from scipy.signal import savgol_filter

# ============ 字体设置 ============
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

# 基本参数
xi_f = 3.6          # 折叠态持续长度
alpha = 7.6         # 解折叠系数

# ===================== 核心变量 =====================
N_val = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]  

R0_Area1 = 1.5                           
R0_Area2 = 20.0                          

k1 = 6.5
k2 = 1.48           
Rtheo_points = 200  
lambda_max = 300.0  

# ===================== 核心函数 =====================
def Lc(f, N):
    contour_length =  N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))
    return contour_length

def MSforce(x):
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def StressOptimization(R0, N, r_val, f_val):
    Re = R0
    r_val = np.asarray(r_val)
    f_val = np.asarray(f_val)

    mask = r_val >= Re
    if not np.any(mask):
        raise ValueError("没有找到 r >= R0 的数据点，请检查 R0 或数据范围。")

    r_selected = r_val[mask]
    lambda_ = r_selected / Re                    
    r2 = lambda_ ** (-0.5) * Re                  

    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])

    sigma = Re * (f1 - lambda_ ** (-1.5) * f2)

    eps = 1e-12
    if len(lambda_) == 0:
        return lambda_, sigma

    if np.abs(lambda_[0] - 1.0) > eps:
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        sigma[0] = 0.0
    return lambda_, sigma

def initial_modulus(R0, N, fit_points=5):
    x_MS = np.linspace(0.01, 0.99, 3000)
    f_MS = MSforce(x_MS)
    r_MS = x_MS * Lc(f_MS, N)              

    lambda_, sigma = StressOptimization(R0, N, r_MS, f_MS)

    lam_fit = lambda_[:fit_points]
    sig_fit = sigma[:fit_points]

    A = (lam_fit - 1.0).reshape(-1, 1)
    G0, _, _, _ = np.linalg.lstsq(A, sig_fit, rcond=None)
    return G0[0]

def calculate_G0_area1(R0, N):
    L_c = N * xi_f                         
    x0 = R0 / L_c
    
    if x0 >= 1:
        raise ValueError(f"计算错误：x0 = {x0} 必须小于 1。")
    
    term1 = x0 / (2 * (1 - x0)**3)
    term2 = 1 / (4 * (1 - x0)**2)
    term3 = 2 * x0
    bracket = term1 + term2 + term3 - 0.25
    coeff = 1.5 * R0
    G0 = coeff * bracket
    return G0

# ============================================================
# ✅【重点修改区】第二种初态（部分解折叠）的精确解析解
# ============================================================
def calculate_G0_area2(R0, N):
    """
    基于第二张图中 Ax^2 + Bx = x0*lambda 的物理模型推导解析解。
    """
    
    # 1. 定义 MS 力-伸长关系函数和其导数
    def f(x):
        if x >= 0.9999: return 1e10 # 避免除零
        return 0.25 * ((1 - x)**(-2) - 1 + 4*x)
    
    def f_prime(x):
        if x >= 0.9999: return 1e10
        return 0.5 * (1 - x)**(-3) + 1

    # 2. 找出 f(x) = k2 的临界平衡点 xc (数值解析值)
    xc = 0.541  # 满足 f(xc) = 1.48 的根
    
    f_prime_xc = f_prime(xc)
    
    # 3. 计算图片2中的常数 A 和 B
    A = (alpha - 1) / 2 * k1 * f_prime_xc
    B = (alpha + 1) / 2 - A * xc
    
    # 4. 确定 x0 = R0 / (N * xi_f)
    x0 = R0 / (N * xi_f)
    if x0 <= 0: return 0
    
    # 5. 解二次方程 A x1^2 + B x1 = x0 得到 x1
    D = np.sqrt(max(0, B**2 + 4 * A * x0))
    x1 = (-B + D) / (2 * A)
    
    # 6. 计算 dx1/d(lambda) 在 lambda=1 时的导数
    dx1 = x0 / D
    
    # 7. 代入 G0 公式
    G0 = 1.5 * R0 * (f(x1) + f_prime(x1) * dx1)
    
    return G0
# ============================================================

def plot_G0_vs_N(N_val, save_dir=None):
    # ===============第一种初态===============
    R0 = R0_Area1  
    G0_val = [initial_modulus(R0, n) for n in N_val]  

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax1.plot(N_val, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)

    N_theo = np.logspace(np.log10(0.5), np.log10(N_val[-1] + 2.0), Rtheo_points)
    G0_theo = [calculate_G0_area1(R0, n) for n in N_theo]
    ax1.plot(N_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical', alpha=0.8, zorder=5)

    ref_x = [N_theo[0], N_theo[-1]]
    ref_y = [G0_theo[-1] * (N_theo[-1] / N_theo[0]), G0_theo[-1]]
    ax1.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5, label=r'$G_0 \propto N^{-1}$', zorder=3)

    ax1.set_xlabel('Number of domains $N$', fontsize=label_fontsize)
    ax1.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax1.set_title(f'Initial modulus $G_0$ vs $N$ ($R_0={R0:.1f}$)', fontsize=title_fontsize, pad=20)
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    ax1.set_xlim(0.5, N_val[-1] + 2.0)
    ax1.set_ylim(0.01, 100.0)

    ax1.tick_params(axis='both', which='major', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width,
                   length=xtick_major_size)
    ax1.tick_params(axis='both', which='minor', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width*0.75,
                   length=xtick_major_size*0.5)
    ax1.minorticks_on()
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'Initial1_G0_vs_N_R0={R0}.png')
        fig1.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"G0 vs N 曲线(情况1)已保存至: {path}")

    # ================第二种初态===========================
    R0 = R0_Area2  
    G0_val = [initial_modulus(R0, n) for n in N_val]  

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    ax2.plot(N_val, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)

    N_theo = np.logspace(np.log10(0.5), np.log10(N_val[-1] + 2.0), Rtheo_points)
    # ✅ 使用修正后的解析解
    G0_theo = [calculate_G0_area2(R0, n) for n in N_theo]  
    ax2.plot(N_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical', alpha=0.8, zorder=5)

    # 参考线 y ∝ N^(-1)，与右端点对齐
    ref_x = [N_theo[0], N_theo[-1]]
    ref_y = [G0_theo[-1] * (N_theo[-1] / N_theo[0]), G0_theo[-1]]
    ax2.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5, label=r'$G_0 \propto N^{-1}$', zorder=3)

    # ✅ 图例与轴标签调整
    ax2.set_xlabel('Number of domains $N$', fontsize=label_fontsize)
    ax2.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax2.set_title(f'Initial modulus $G_0$ vs $N$ ($R_0={R0:.1f}$)', fontsize=title_fontsize, pad=20)
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    ax2.set_xlim(0.1, N_val[-1] + 1.0)
    ax2.set_ylim(30.0, 1000.0)

    ax2.tick_params(axis='both', which='major', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width,
                   length=xtick_major_size)
    ax2.tick_params(axis='both', which='minor', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width*0.75,
                   length=xtick_major_size*0.5)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'Initial2_G0_vs_N_R0={R0}.png')
        fig2.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"G0 vs N 曲线(情况2修正后)已保存至: {path}")

# ========================== 主函数 ==========================
def main():
    print("=" * 80)
    print("开始生成初始模量 G0 随 N 变化的比较图...")
    print("=" * 80)
    
    data_dir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_Constant"  
    output_dir = data_dir  
    
    print("\n开始生成 G0 随 N 变化曲线...")
    plot_G0_vs_N(N_val, save_dir=output_dir)
    
    print("=" * 80)
    print("所有曲线生成完成。")
    print("=" * 80)

if __name__ == "__main__":
    main()