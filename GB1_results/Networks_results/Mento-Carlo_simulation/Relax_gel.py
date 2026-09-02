import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq  # 新增导入

# ============ 字体与绘图风格设置 ============
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
ytick_left = True
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
kR = 2.68          
xi_f = 3.6         
alpha = 7.6        
k1 = 6.5
k2 = 1.50

# ===================== 基础物理函数 =====================
def phi(x):
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def fc(x):
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def fcp(x):
    return 0.5 * (1 - x)**(-3) + 1.0

def f_MS(x):
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def Lc_of_f(f, N):
    return N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))

# ===================== 理论解 G0 =====================
# 以下函数使用的是：3-chain 拓扑 + 单链的力-拉伸关系
def G0_partially_unfolded(N):
    """
    参考图片中 Initial modulus: partially unfolded 公式计算 G0
    G0 = (ρ * kB T * R0 / lp) * 3/2 * [ f(x1) + f'(x1) * x1' ]
    其中 3ρ = n = 1, kB T = 1, lp = 1，所以 ρ = 1/3
    故 G0 = (1/3 * 1 * R0 / 1) * 3/2 * [ ... ] = 0.5 * R0 * [ ... ]
    """
    # 初始参数
    R0 = kR * np.sqrt(N)
    x0 = R0 / (N * xi_f)
    
    # 求解 xc，使得 f(xc) = k2
    # 使用 brentq 在 (0, 1) 区间内寻找根
    def func_to_solve(x):
        return f_MS(x) - k2
    xc = brentq(func_to_solve, 0, 1)
    
    # 计算 f'(xc)
    df_dx_c = fcp(xc)
    
    # 计算系数 A 和 B
    A = (alpha - 1) / 2 * k1 * df_dx_c
    B = (alpha + 1) / 2 - A * xc
    
    # 计算 x1 和 x1'
    term = np.sqrt(B**2 + 4 * A * x0)
    x1 = (-B + term) / (2 * A)
    x1_prime = x0 / term
    
    # 计算最终 G0
    G0 = 0.5 * R0 * (f_MS(x1) + fcp(x1) * x1_prime)
    return G0


def G0_3chain_folded(N, kR=kR, xi_f=xi_f):
    R0 = kR * np.sqrt(N)
    L = N * xi_f
    x0 = R0 / L
    if x0 >= 1: 
        return np.inf
    
    term1 = x0 / (2 * (1 - x0)**3)
    term2 = 1 / (4 * (1 - x0)**2)
    term3 = 2 * x0
    bracket = term1 + term2 + term3 - 0.25
    
    return R0 * 0.5 * bracket

# ===================== 3-chain 拓扑 + WlC 模型求解 =====================

def G0_3chain_WLC(N, points=5000):
    """
    严格参考图片中 3-chain topology 公式计算 G0
    G0(L) = (nL/2) * ∫ x [x f_c'(x) + f_c(x)] p(x; L) dx
    L = N * xi_f (固定轮廓长度)
    p(x; L) = (1/Z) x^2 exp[-L * Phi(x)]
    此处 n=1, kB T=1, lp=1
    """
    L = N * xi_f
    
    # 构造高精度积分网格，避开 x=1
    x_grid = np.concatenate([
        np.linspace(0, 0.9, 1000),
        1 - np.logspace(-3, -9, points)
    ])
    x_grid = x_grid[x_grid < 1]
    
    # 计算自由能 F_c(x; L) = L * Phi(x)
    Fc = L * phi(x_grid)
    f = fc(x_grid)
    f_prime = fcp(x_grid)
    
    # 概率分布 p(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = 2 * np.log(np.clip(x_grid, 1e-300, None)) - Fc
        log_p -= np.max(log_p)
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
    
    Z = np.trapezoid(p_un, x_grid)
    p = p_un / Z
    
    # 积分计算 G0
    integrand = x_grid * (x_grid * f_prime + f) * p
    G0 = 0.5 * L * np.trapezoid(integrand, x_grid)
    return G0

# ===================== 3-chain 拓扑 + 可变的轮廓长度 模型求解 =====================

def G0_3chain_components(N, x_grid):
    f = f_MS(x_grid)
    Lc = Lc_of_f(f, N)
    r = x_grid * Lc
    
    dr_dx = np.gradient(r, x_grid)
    df_dx = np.gradient(f, x_grid)
    df_dr = df_dx / dr_dx
    
    Fc = cumulative_trapezoid(f, r, initial=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = 2 * np.log(np.clip(r, 1e-300, None)) - Fc
        log_p -= np.max(log_p)
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
    
    Z = np.trapezoid(p_un, r)
    p = p_un / Z
    
    return r, f, df_dr, Fc, p

def G0_3chain_quad(N, points=5000):
    x_grid = np.concatenate([
        np.linspace(0, 0.9, 1000),
        1 - np.logspace(-3, -9, 4000)
    ])
    r, f, df_dr, Fc, p = G0_3chain_components(N, x_grid)
    
    integrand = 0.5 * r * (r * df_dr + f) * p
    G0 = np.trapezoid(integrand, r)
    return G0

# ===================== 可视化函数 =====================
def plot_G0_3chain(N_vals, save_dir=None):
    G0_quad_vals = []
    G0_theory_vals = []
    G0_partial_vals = []
    G0_WLC_vals = []
    
    for N in N_vals:
        G0_quad_vals.append(G0_3chain_quad(N))
        G0_theory_vals.append(G0_3chain_folded(N))
        G0_partial_vals.append(G0_partially_unfolded(N))
        G0_WLC_vals.append(G0_3chain_WLC(N)) # 新增计算
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(1, max(N_vals))
    ax.set_ylim(1, 10)

    # 3-chain 拓扑 + 可解折叠链 的数值积分解
    ax.plot(N_vals, G0_quad_vals, 'o', label='Num:3-chain + unfoldable', markersize=10, color='blue', zorder=6)
    # 3-chain 拓扑 + 完全折叠
    ax.plot(N_vals, G0_theory_vals, 'o-', color='black',  markersize=10, linewidth=3, label=f'$R_0=2.68 \\sqrt{{N}}$', zorder=5)
    # 3-chain 拓扑 + 部分展开
    ax.plot(N_vals, G0_partial_vals, 'o-', color='orange', linewidth=4, label='Partially Unfolded', markersize=10, zorder=4, markeredgecolor='black')
    # 3-chain 拓扑 + WLC 模型 的数值积分解
    ax.plot(N_vals, G0_WLC_vals, 'o-', color='green', linewidth=3, markersize=10, label='3-chain + WLC (fixed $L=N\\xi_f$)', zorder=5)
    
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=3, label='Rubber Modulus')
    
    ax.set_xlabel('Number of domains $N$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / n k_B T$', fontsize=label_fontsize)
    ax.set_title('3-chain topology', fontsize=title_fontsize, pad=20)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    # ax.legend(fontsize=legend_fontsize*0.8, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, 'G0_3chain_vs_N.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"3-chain图已保存至: {path}")


def plot_distribution_r(N, save_dir=None):
    """
    可视化给定 N 下的分布函数 p(r; N) 及累积分布函数 CDF
    """
    x_grid = np.concatenate([
        np.linspace(0, 0.9, 1000),
        1 - np.logspace(-3, -9, 4000)
    ])
    r, f, df_dr, Fc, p = G0_3chain_components(N, x_grid)
    
    # 计算累积分布函数 CDF = ∫ p dr
    cdf = cumulative_trapezoid(p, r, initial=0)
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    ax2 = ax.twinx()  # 双y轴
    
    # 绘制 PDF
    ax.plot(r, p, color='blue', linewidth=lines_linewidth, label=f'PDF ($N={N}$)')
    ax.set_xlabel(r'$r$', fontsize=label_fontsize)
    ax.set_ylabel(r'$p(r | N)$', fontsize=label_fontsize, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # 绘制 CDF
    ax2.plot(r, cdf, color='red', linestyle='--', linewidth=lines_linewidth, label=f'CDF ($N={N}$)')
    ax2.set_ylabel(r'$C(r | N)$', fontsize=label_fontsize, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.axvline(x=1.95, color='purple', linestyle='--', linewidth=3, label='$r=1.95$')
    
    # 标题
    ax.set_title(f'Distribution of $r$ ($N = {N}$)', fontsize=title_fontsize, pad=20)
    
    # 网格与图例（合并图例）
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置刻度等格式
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, left=ytick_left)
    ax2.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    # 限制显示范围，聚焦主要区域
    p_max_idx = np.argmax(p)
    r_max = 10.0  # 根据物理意义设置 r 的最大值
    ax.set_xlim(0, r_max)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(-0.05, 1.05)  # CDF从0到1
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, f'distribution_r_N{N}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"分布图已保存至: {path}")
    
    plt.show()
    return fig, ax


# ===================== 主程序入口 =====================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/Mento-Carlo_simulation'
    os.makedirs(output_dir, exist_ok=True)
    
    N_vals = np.arange(1, 21, 1)
    
    plot_G0_3chain(N_vals, save_dir=output_dir)

    # 可视化分布函数 p(r; N) 对于单个 N 值
    N_single = 1
    plot_distribution_r(N_single, save_dir=output_dir)

if __name__ == "__main__":
    main()