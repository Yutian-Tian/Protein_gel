"""
绘制初始模量 G₀ 随 N 的变化，并与理论公式对比
R0与N存在关系 R0 = kR * N**0.5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import brentq

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
k1 = 7.5
k2 = 1.5
kR = 2.68           # 初始首末端距离 R0 = kR * N**0.5

N_Area2_fixed = 8.0    # 部分解折叠模型中固定使用的 N
N_vals = np.linspace(1.0, 10.0, 10)  # 数值优化离散点
Rtheo_points = 200

# ===================== 基础物理函数 =====================
def Lc(f, N):
    return N * xi_f * (0.5 * (alpha + 1) + 0.5 * (alpha - 1) * np.tanh(k1 * (f - k2)))

def MSforce(x):
    return np.where(x < 0.99,
                    0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                    np.inf)

def R0ms(N):
    L = N*xi_f
    Rms = np.sqrt(2*L*(1 - 1/L*(1 - np.exp(-L))))
    return Rms

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
    基于第二张图中 Ax^2 + Bx = x0*lambda 的物理模型推导解析解。
    """
    
    # 1. 定义 MS 力-伸长关系函数和其导数
    def f(x):
        if x >= 0.9999: return 1e10 
        return 0.25 * ((1 - x)**(-2) - 1 + 4*x)
    
    def f_prime(x):
        if x >= 0.9999: return 1e10
        return 0.5 * (1 - x)**(-3) + 1

    # 2. 【修改此处】使用数值方法严格求解 f(xc) = k2 = 1.48 的根
    xc = brentq(lambda x: f(x) - k2, 0.1, 0.9)  
    
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

# ===================== 新增：临界（精确）理论 G0 计算函数 =====================
def calculate_G0_critical(R0, N):
    """
    基于图片公式精确计算理论初始模量 G₀。
    该公式相比 partial unfolding (Area 2) 公式，
    完整保留了求导项中的 x₀/2 部分。
    公式：G₀ = 1.5 * R₀ * [f(x₁) + f'(x₁) * (dx₁/dλ + x₀/2)]
    其中 x₀ = R₀ / (N * ξ_f)
    """
    
    # 1. 定义 MS 力-伸长关系函数和其导数
    def f(x):
        if x >= 0.9999: return 1e10 
        return 0.25 * ((1 - x)**(-2) - 1 + 4*x)
    
    def f_prime(x):
        if x >= 0.9999: return 1e10
        return 0.5 * (1 - x)**(-3) + 1

    # 2. 求解临界点 xc，使得 f(xc) = k2
    xc = brentq(lambda x: f(x) - k2, 0.1, 0.9)  
    f_prime_xc = f_prime(xc)
    
    # 3. 计算图片中的参数 A 和 B
    A = (alpha - 1) / 2 * k1 * f_prime_xc
    B = (alpha + 1) / 2 - A * xc
    
    # 4. 确定 x0
    x0 = R0 / (N * xi_f)
    if x0 <= 0: return 0
    
    # 5. 解一元二次方程 A * x1^2 + B * x1 - x0 = 0 得到 x1
    D = np.sqrt(max(0, B**2 + 4 * A * x0))
    x1 = (-B + D) / (2 * A)
    
    # 6. 计算 x(λ) 对 λ 在 λ=1 处的导数 dx1/dλ
    # 由 x(λ) = (-B + sqrt(B**2 + 4*A*x0*λ))/(2*A) 求导可得
    dx1 = x0 / D
    
    # 7. 【核心修正】代入图片中完整的 G0 公式
    # (注意：此处比 area2 多出了 x0/2 这一项)
    G0 = 1.5 * R0 * (f(x1) + f_prime(x1) * (dx1 + x0 / 2))
    
    return G0

# ===================== 针对部分打开初态：使用渐近，理论 G0 计算函数 =====================
def calculate_InitModulus_area2(R0, N):
    """
    基于 Area 2 (部分解折叠) 的理论近似解析解求初始模量 G0
    严格依据图片中 r(f) = a + bΔf + c(Δf)^2 的近似推导确定 a、b、c
    """
    
    # 1. 计算 L_c(f) 近似展开系数
    # L_c(f) ≈ L0 + L1 * Δf
    L0 = N * xi_f * (alpha + 1) / 2
    L1 = N * xi_f * (alpha - 1) / 2 * k1
    
    # 2. 计算 x_MS(f) 近似展开系数
    # x_MS(f) ≈ X0 + X1 * Δf
    # (k_B T = 1, l_p = 1)
    X0 = 1 - 1 / np.sqrt(k2)
    X1 = 1 / (2 * k2**1.5)
    
    # 3. 确定二次多项式 r(f) = a + b*Δf + c*(Δf)^2 中的系数
    a = X0 * L0
    b = X1 * L0 + X0 * L1
    c = X1 * L1
    
    # 4. 计算 f(R0) 和 f'(R0) (逆函数求解)
    # 防止数值误差导致 delta < 0，使用 max(0, ...) 过滤
    discriminant = max(0, b**2 - 4 * c * (a - R0))
    sqrt_disc = np.sqrt(discriminant)
    
    # 避免分母为 0 的极端情况
    if sqrt_disc < 1e-12:
        return 0.0
        
    f_R0 = k2 + (-b + sqrt_disc) / (2 * c)
    f_prime_R0 = 1 / sqrt_disc
    
    # 5. 计算 G0 (取 rho*kB_T/l_p = 1)
    G0 = R0 * (R0 / 2 * f_prime_R0 + 1.5 * f_R0)
    
    return G0

# ===================== 可视化函数 =====================
def plot_G0_vs_N_area1(N_vals, kR=0.5, save_dir=None):
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

# ===================== 修改后的绘图函数 =====================
def plot_G0_vs_N_area2(N_vals, kR=2.68, save_dir=None):
    """
    绘制第二种情况 (部分解折叠) G₀ 随 N 变化
    横轴改为 N，与 plot_G0_vs_N_area1 保持一致，并将 Area 1 理论解以蓝实线绘制在同一张图上
    """
    
    # 根据 N_vals 列表计算相应的 R0 列表
    # R0_val = [kR * n**0.5 for n in N_vals]
    R0_val = [R0ms(n) for n in N_vals]
    G0_val = [initial_modulus_num(r0, n) for r0, n in zip(R0_val, N_vals)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    # 1. 数值优化解 (Area 2)
    ax.plot(N_vals, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)
    
    # 生成连续的 N_theo 作为 x 轴
    N_theo = np.linspace(1.0, 10.5, Rtheo_points)

    # 2. 新增：理论解析解 (Area 1) - 蓝实线
    G0_theo_area1 = [calculate_G0_area1(R0ms(n), n) for n in N_theo]
    ax.plot(N_theo, G0_theo_area1, '-', color='blue', linewidth=lines_linewidth, label='Fully Folded', alpha=0.8, zorder=5)
    
    # 3. 理论解析解 (Area 2) - 原图的理论线
    G0_theo_area2 = [calculate_G0_area2(R0ms(n), n) for n in N_theo]
    ax.plot(N_theo, G0_theo_area2, '-', color='orange', linewidth=lines_linewidth, label='Partially Unfolded', alpha=0.8, zorder=5)

    # G0_theo_critical = [calculate_G0_critical(R0ms(n), n) for n in N_theo]
    # ax.plot(N_theo, G0_theo_critical, '-', color='red', linewidth=lines_linewidth, label='Critical case', alpha=0.8, zorder=5)

    # G0_theo_Initial = [calculate_InitModulus_area2(R0ms(n), n) for n in N_theo]
    # ax.plot(N_theo, G0_theo_Initial, '-', color='green', linewidth=lines_linewidth, label='Initial Modulus', alpha=0.8, zorder=5)

    ax.set_xlabel('Number of domains $N$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Initial modulus $G_0$ vs. $N$', fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.set_xlim(1.0, 10.5)
    ax.set_ylim(0.0, 20.0) 
    
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'Initial2_G0_vs_N_kr={kR}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Area2 对比曲线已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_N0.5'
    
    # 绘制基于 R0 = kR*sqrt(N) 的标度律对比 (以 N 为横坐标)
    plot_G0_vs_N_area1(N_vals, save_dir=output_dir)
    plot_G0_vs_N_area2(N_vals, save_dir=output_dir)

if __name__ == "__main__":
    main()