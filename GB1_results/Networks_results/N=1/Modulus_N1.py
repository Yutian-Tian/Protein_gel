import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
from scipy.integrate import cumulative_trapezoid

# ============ 字体与绘图风格设置 (保持不变) ============
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
mu = 7.6        
Delta_E = 11.9 

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分"""
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def f_MS(x):
    """微观力解析计算 (beta=1, lp=1)"""
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def fcp(x):
    """微观力对 x 的解析导数"""
    x = np.clip(x, 0, 0.999999)
    return 0.5 * (1 - x)**(-3) + 1.0

def Lc_of_n(n, N, xi_f):
    """
    修正：轮廓长度 Lc 必须严格依赖状态变量 n (图片第一张图)，而非力 f
    """
    return N * xi_f * (1 + n*(mu - 1))

def U_of_n(n):
    """图片中显式定义的势能项 U(n)"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

# ===================== 模型 1：固定轮廓长度 WLC 模型 (绿色线基线) =====================
def G0_3chain_WLC(N, xi_f, points=5000):
    """固定轮廓长度 L = N * xi_f 的 WLC 模型（保持不变）"""
    L = N * xi_f
    x_grid = np.concatenate([np.linspace(0, 0.9, 1000), 1 - np.logspace(-3, -9, points)])
    x_grid = x_grid[x_grid < 1]
    
    Fc = L * phi(x_grid)
    f = f_MS(x_grid)
    f_prime = fcp(x_grid)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = 2 * np.log(np.clip(x_grid, 1e-300, None)) - Fc
        log_p -= np.max(log_p)
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
    
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    Z = trapz_func(p_un, x_grid)
    p = p_un / Z
    
    integrand = x_grid * (x_grid * f_prime + f) * p
    G0 = 0.5 * L * trapz_func(integrand, x_grid)
    return G0

# ===================== 可解折叠模型：严格双重积分 (n 和 r) 蓝色线 =====================
def G0_3chain_quad(N, xi_f, n_points=50, r_points=2000):
    n_grid = np.linspace(0, 1, n_points)
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    
    log_p_list = []
    r_grid_list = []
    f_val_list = []
    df_dr_list = []
    
    max_log_p_global = -np.inf
    
    # 第一步：计算并存储所有局部数据，寻找全局最大值
    for n in n_grid:
        Lc = Lc_of_n(n, N, xi_f)
        U_n = U_of_n(n)
        r_grid = np.linspace(0, Lc * 0.9999, r_points)
        x_grid = r_grid / Lc
        
        f_val = f_MS(x_grid)
        f_prime_x = fcp(x_grid)
        df_dr = f_prime_x / Lc
        F_d = Lc * phi(x_grid) + U_n
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = 2 * np.log(np.clip(r_grid, 1e-300, None)) - F_d
            log_p = np.nan_to_num(log_p, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        
        log_p_list.append(log_p)
        r_grid_list.append(r_grid)
        f_val_list.append(f_val)
        df_dr_list.append(df_dr)
        max_log_p_global = max(max_log_p_global, np.max(log_p))
        
    # 第二步：使用全局最大值统一归一化，并计算积分
    Z_total = 0.0
    E_total = 0.0
    dn = n_grid[1] - n_grid[0]
    
    for idx, n in enumerate(n_grid):
        log_p = log_p_list[idx] - max_log_p_global  # 统一归一化
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
        
        r_grid = r_grid_list[idx]
        f_val = f_val_list[idx]
        df_dr = df_dr_list[idx]
        
        Z_n = trapz_func(p_un, r_grid)
        Z_total += Z_n * dn
        
        integrand = 0.5 * (r_grid**2 * df_dr + r_grid * f_val) * p_un
        E_n = trapz_func(integrand, r_grid)
        E_total += E_n * dn
        
    return E_total / Z_total

# ===================== 精确计算边缘分布 =====================
def calculate_pn(N, xi_f, n_points=100, r_points=2000):
    n_grid = np.linspace(0, 1, n_points)
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    
    Z_n_vals = np.zeros_like(n_grid)
    max_log_p_global = -np.inf  # 记录全局最大对数概率
    
    # 第一步：计算所有 n 下的未归一化对数概率，并记录全局最大值
    log_p_list = []
    for idx, n in enumerate(n_grid):
        Lc = Lc_of_n(n, N, xi_f)
        U_n = U_of_n(n)
        r_grid = np.linspace(0, Lc * 0.9999, r_points)
        x_grid = r_grid / Lc
        F_d = Lc * phi(x_grid) + U_n
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = 2 * np.log(np.clip(r_grid, 1e-300, None)) - F_d
            log_p = np.nan_to_num(log_p, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        
        log_p_list.append(log_p)
        max_log_p_global = max(max_log_p_global, np.max(log_p))
    
    # 第二步：使用全局最大值进行统一归一化，然后计算 Z_n
    for idx, n in enumerate(n_grid):
        log_p = log_p_list[idx] - max_log_p_global  # <--- 统一归一化
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
        Lc = Lc_of_n(n, N, xi_f)
        r_grid = np.linspace(0, Lc * 0.9999, r_points)
        Z_n_vals[idx] = trapz_func(p_un, r_grid)
        
    Z_total = trapz_func(Z_n_vals, n_grid)
    p_n = Z_n_vals / Z_total
    print(sum(n_grid * p_n * (n_grid[1] - n_grid[0])))  # 输出期望值 <n>

    return n_grid, p_n

# ===================== 可视化函数 =====================
def plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=None):
    G0_quad_vals = []
    G0_WLC_vals = []
    
    print(f"正在计算 N={N_single} 下的数据 (严格双重积分)...")
    for xi_f in xi_f_vals:
        G0_quad_vals.append(G0_3chain_quad(N_single, xi_f))
        G0_WLC_vals.append(G0_3chain_WLC(N_single, xi_f))
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(min(xi_f_vals), max(xi_f_vals))
    ax.set_ylim(2.0, 100.0)
    ax.set_ylim(2.0, 10.0)
    
    # 橙色线：完全按图片公式的双重积分模型
    ax.plot(xi_f_vals, G0_quad_vals, 's', color='blue', 
            linewidth=3, markersize=7, label='Exact Integration', zorder=6)
    
    # 绿色线：固定轮廓长度 WLC
    ax.plot(xi_f_vals, G0_WLC_vals, 'o', color='green', 
            linewidth=3, markersize=7, label='WLC', zorder=5)

    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=3, label='$G_0^{Rubber}=$3.0', zorder=1)
    
    ax.set_xlabel('Folded length $\\xi_\\mathrm{f}$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / \\rho k_\\mathrm{B} T$', fontsize=label_fontsize)
    ax.set_title(f'Initial Modulus', fontsize=title_fontsize, pad=20)
    
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, f'G0_vs_xi_f_N{N_single}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"图表已保存至: {path}")

# ===================== 可视化边缘分布p(n) =====================
def plot_pn_vs_n(N_single, xi_f, save_dir=None):
    """
    可视化边缘分布 p(n)
    """
    n_grid, p_n = calculate_pn(N_single, xi_f)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('log')
    
    # 绘制 p(n)
    ax.plot(n_grid, p_n, '-', color='purple', linewidth=4, label=f'$\\xi_\\mathrm{{f}}={xi_f}$')
    # 可以用散点强调
    ax.plot(n_grid, p_n, 'o', color='purple', markersize=6, markerfacecolor='white')
    
    ax.set_xlabel('State variable $n$', fontsize=label_fontsize)
    ax.set_ylabel('Probability density $p(n)$', fontsize=label_fontsize)
    ax.set_title(f'Marginal Distribution ($N={N_single}$)', fontsize=title_fontsize, pad=20)
    
    ax.set_xlim(0, 1)
    # 留出一点 y 轴余量
    #ax.set_ylim(0, 20.0)
    
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize*0.8, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'pn_distribution_N{N_single}_xi{xi_f}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"分布图已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures'
    os.makedirs(output_dir, exist_ok=True)
    
    N_single = 1
    xi_f_vals = np.linspace(1.0, 100.0, 100)
    print(G0_3chain_WLC(N_single, 3.6))
    print(G0_3chain_quad(N_single, 3.6))
    
    plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=output_dir)
    plot_pn_vs_n(N_single, xi_f=3.6, save_dir=output_dir)
    plot_pn_vs_n(N_single, xi_f=50.0, save_dir=output_dir)

if __name__ == "__main__":
    main()