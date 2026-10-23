"""
目的：计算精确的本构曲线，与唯象模型进行对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os

# ============ 字体与绘图风格设置 (严格保持前文风格) ============
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
lines_markersize = 10

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
mu = 7.6        
k1 = 6.5
k2 = 1.50
Delta_E = 11.7 # 能量势垒，可调整

## ===========计算参数=============
R0 = 2.0   # 唯象模型的初始端距
lam_max = 1.5
 

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分"""
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def f_MS(x):
    """微观力解析式 (beta=1, lp=1)"""
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def Lc_of_n(n, N, xi_f):
    """方法二：轮廓长度依赖状态变量 n"""
    return N * xi_f * (1 + n*(mu - 1))

def Lc_of_f(f, N, xi_f):
    """方法一：轮廓长度依赖瞬时力 f (绝热近似)"""
    return N * xi_f * (0.5*(mu + 1) + 0.5*(mu - 1)*np.tanh(k1*(f - k2)))

def U_of_n(n):
    """图片中定义的势能项 U(n)"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

# ===================== 方法一：唯象三链网络模型 =====================
def calculate_stress_method1(lambda_vals, N, xi_f, R0=10.0, x_points=2000):
    """
    方法一：直接使用力-拉伸曲线代入三链网络的拓扑结构。
    不再采用分布函数再积分的方法。
    参数:
        lambda_vals: 宏观拉伸比数组
        N: domain 数量
        xi_f: 折叠态持续长度
        R0: 初始首末端距离 (默认 10.0)
        x_points: 微观拉伸比网格点数
    返回:
        sigma_lambda: 应力数组
    """
    # 生成微观力-伸长曲线 (r, f)
    x_grid = np.linspace(0, 0.999, x_points)
    f_grid = f_MS(x_grid)
    Lc_grid = Lc_of_f(f_grid, N, xi_f)
    r_grid = x_grid * Lc_grid          # 端距 r 随 x 单调递增

    sigma_lambda = np.zeros_like(lambda_vals)
    for idx, lam in enumerate(lambda_vals):
        # 拉伸方向端距
        r1 = lam * R0
        # 横向端距（两个方向相同）
        r2 = lam**(-0.5) * R0

        # 从 f-r 曲线插值得到力
        f1 = np.interp(r1, r_grid, f_grid, left=f_grid[0], right=f_grid[-1])
        f2 = np.interp(r2, r_grid, f_grid, left=f_grid[0], right=f_grid[-1])

        # 三链网络应力公式：σ = 1/3 * (f1 - λ^{-1.5} * f2)
        sigma_lambda[idx] = R0*(f1 - lam**(-1.5) * f2) / 3.0

    return sigma_lambda

# ===================== 方法二：图片严格双重积分模型 =====================
def calculate_stress_method2(lambda_vals, N, xi_f, n_points=50, x_points=1000):
    """
    方法二：严格根据图片定义
    对状态变量 n 积分，且对每个 n 状态下的端距 r 积分。
    微观自由能 F_d = F_WLC + U(n)
    """
    n_grid = np.linspace(0, 1, n_points)
    x_grid = np.linspace(1e-5, 0.999, x_points)
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    
    sigma_lambda = np.zeros_like(lambda_vals)
    
    for idx, lam in enumerate(lambda_vals):
        Lambda_x, dLambda_x = lam, 1.0
        Lambda_y, dLambda_y = lam**(-0.5), -0.5 * lam**(-1.5)
        Lambda_z, dLambda_z = lam**(-0.5), -0.5 * lam**(-1.5)
        dirs = [(Lambda_x, dLambda_x), (Lambda_y, dLambda_y), (Lambda_z, dLambda_z)]
        
        Z_total = 0.0
        sigma_integral = 0.0
        dn = n_grid[1] - n_grid[0]
        
        for n in n_grid:
            Lc = Lc_of_n(n, N, xi_f)
            U_val = U_of_n(n)
            
            F_wlc = Lc * phi(x_grid)
            F_d = F_wlc + U_val
            
            # 径向概率分布 p_r(r,n) ~ r^2 * exp(-F_d)
            r_grid = x_grid * Lc
            weight_r = (r_grid**2) * np.exp(-F_d) * Lc # dr = Lc dx
            
            Z_n = trapz_func(weight_r, x_grid)
            Z_total += Z_n * dn
            
            dir_sum = 0.0
            for Lam, dLam in dirs:
                x_eff = np.clip(x_grid * Lam, 0, 0.999999)
                f_val = f_MS(x_eff)
                dir_sum += f_val * r_grid * dLam
            
            integrand = (dir_sum / 3.0) * weight_r
            sigma_integral += trapz_func(integrand, x_grid) * dn
            
        sigma_lambda[idx] = sigma_integral / Z_total
        
    return sigma_lambda

# ===================== 参考线：固定轮廓长度 WLC =====================
def calculate_stress_WLC(lambda_vals, N, xi_f, x_points=2000):
    """无解折叠效应的参考线"""
    L = N * xi_f
    x_grid = np.linspace(1e-5, 0.999, x_points)
    F_grid = L * phi(x_grid)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = 2 * np.log(np.clip(x_grid, 1e-300, None)) - F_grid
        log_p -= np.max(log_p)
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
        
    trapz_func = getattr(np, 'trapezoid', np.trapz)
    Z = trapz_func(p_un, x_grid)
    p = p_un / Z
    
    sigma_lambda = np.zeros_like(lambda_vals)
    for idx, lam in enumerate(lambda_vals):
        Lambda_x, dLambda_x = lam, 1.0
        Lambda_y, dLambda_y = lam**(-0.5), -0.5 * lam**(-1.5)
        Lambda_z, dLambda_z = lam**(-0.5), -0.5 * lam**(-1.5)
        
        sum_f = 0.0
        for Lam, dLam in [(Lambda_x, dLambda_x), (Lambda_y, dLambda_y), (Lambda_z, dLambda_z)]:
            x_eff = np.clip(x_grid * Lam, 0, 0.999999)
            f_eff = f_MS(x_eff)
            sum_f += f_eff * x_grid * dLam
            
        sigma_lambda[idx] = trapz_func(sum_f / 3.0 * p, x_grid)
        
    return sigma_lambda

# ===================== 可视化函数 =====================
def plot_comparison(N_single, xi_f, save_dir=None):
    lambda_vals = np.linspace(1.0, lam_max, 100) # 拉伸比范围
    
    print(f"正在计算 N={N_single}, xi_f={xi_f} 的本构曲线...")
    sig_m1 = calculate_stress_method1(lambda_vals, N_single, xi_f, R0=R0)
    sig_m2 = calculate_stress_method2(lambda_vals, N_single, xi_f)
    sig_wlc = calculate_stress_WLC(lambda_vals, N_single, xi_f)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    # 绘制三条曲线
    ax.plot(lambda_vals, sig_m1, 's--', color='darkorange', 
            linewidth=3, markersize=7, label='Method 1: Phenomenological 3-chain', zorder=6)
    ax.plot(lambda_vals, sig_m2, 'o-', color='royalblue', 
            linewidth=3, markersize=7, label='Method 2: Strict Double Integral', zorder=5)
    ax.plot(lambda_vals, sig_wlc, '^:', color='green', 
            linewidth=3, markersize=7, label='Reference: Fixed WLC', zorder=4)
    
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('Stretch Ratio $\\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Nominal Stress $\\sigma$', fontsize=label_fontsize)
    ax.set_title(f'Constitutive Curve Comparison ($N={N_single}$, $\\xi_f={xi_f}$)', 
                 fontsize=title_fontsize, pad=20)
    
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize*0.6, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, f'Constitutive_Curve_N{N_single}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"图表已保存至: {path}")
    
    plt.show()

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 固定单 domain 和某个 xi_f 值进行对比
    N_single = 1
    xi_f_val = 3.6
    
    plot_comparison(N_single, xi_f_val, save_dir=output_dir)