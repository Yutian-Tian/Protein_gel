import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid, simpson

# ============ 字体与绘图风格设置 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 28
label_fontsize = 28
tick_fontsize = 24
legend_fontsize = 24
legend_title_fontsize = 24

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 3
lines_markersize = 10

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
mu = 7.6            
Delta_E_mean = 11.7  # 均值
Delta_E_std = 1.7    # 标准差
n_samples = 100      # 蒙特卡洛采样次数
k1 = 6.5
k2 = 1.50

# ===================== 基础物理函数 =====================
def phi(x):
    x = np.clip(x, 0, 0.999999)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def fc(x):
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def Lc_of_n(n):
    return xi_f * (1 + (mu - 1) * n)

def F_WLC(r, n):
    Lc = Lc_of_n(n)
    x = r / Lc
    x_clipped = np.clip(x, 0, 0.9999)
    res = (Lc / 4) * phi(x_clipped)
    return np.where(x >= 0.9999, 1e10, res)

def U_n(n, Delta_E):
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

def F_d(r, n, Delta_E):
    return F_WLC(r, n) + U_n(n, Delta_E)

# ===================== 四种计算 F_eff(r) 的方法（引入 Delta_E 参数） =====================
def calc_method1():
    """方法一：唯象力学积分"""
    x_vals = np.linspace(0, 0.999, 1000)
    f_vals = fc(x_vals)
    n_vals = 0.5 * (1 + np.tanh(k1 * (f_vals - k2)))
    r_vals = x_vals * Lc_of_n(n_vals)
    F_eff1 = cumulative_trapezoid(f_vals, r_vals, initial=0)
    return r_vals, F_eff1

def calc_method2(r_grid, Delta_E):
    """方法二：二态离散求和"""
    F_eff2 = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= xi_f * mu * 0.99:
            F_eff2[i] = 1e10
            continue
        F_folded = F_d(r, 0.0, Delta_E)
        F_unfolded = F_d(r, 1.0, Delta_E)
        F_min = min(F_folded, F_unfolded)
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff2[i] = 1e10
            continue
        Z_sum = np.exp(-(F_folded - F_min)) + np.exp(-(F_unfolded - F_min))
        Z_sum = max(Z_sum, 1e-300)
        F_eff2[i] = F_min - np.log(Z_sum)
    return F_eff2 + Delta_E

def calc_method3_continuous(r_grid, Delta_E):
    """方法三：严格连续积分（对 n 从 0 到 1 连续积分）"""
    F_eff3 = np.zeros_like(r_grid)
    n_star_vals = np.zeros_like(r_grid)
    
    for i, r in enumerate(r_grid):
        if r >= xi_f * mu * 0.999:
            F_eff3[i] = 1e10
            n_star_vals[i] = 1.0
            continue
            
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        res = minimize_scalar(lambda n: F_d(r, n, Delta_E), bounds=(n_min, 1.0), method='bounded')
        F_min = res.fun
        n_star_vals[i] = res.x
        
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff3[i] = 1e10
            continue
            
        n_vals = np.linspace(0, 1, 1000)
        F_d_vals = F_d(r, n_vals, Delta_E)
        integrand = np.exp(-(F_d_vals - F_min))
        integral = simpson(integrand, x=n_vals)
        integral = max(integral, 1e-300)
        F_eff3[i] = F_min - np.log(integral)
        
    return F_eff3 - F_eff3[0], n_star_vals

def calc_method4_adiabatic(r_grid, Delta_E):
    """方法四：绝热/零点近似"""
    F_eff4 = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= xi_f * mu * 0.999:
            F_eff4[i] = 1e10
            continue
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1) + 1e-4)
        res = minimize_scalar(lambda n: F_d(r, n, Delta_E), bounds=(n_min, 1.0), method='bounded')
        F_min = res.fun
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff4[i] = 1e10
            continue
        F_eff4[i] = F_min
    return F_eff4 + Delta_E

# ===================== 可视化函数（引入蒙特卡洛平均） =====================
def plot_comparison(output_dir):
    r_max = 0.98 * xi_f * mu 
    r_grid = np.linspace(0, r_max, 1000)
    
    # 获取 Method 1 的 r 网格（方法1不依赖 Delta_E，计算一次即可）
    r_m1, F_m1_single = calc_method1()
    
    # 初始化用于存储 100 次采样结果的数组
    F_m1_samples = np.zeros((n_samples, len(r_m1)))
    F_m2_samples = np.zeros((n_samples, len(r_grid)))
    F_m3_samples = np.zeros((n_samples, len(r_grid)))
    F_m4_samples = np.zeros((n_samples, len(r_grid)))
    n_star_samples = np.zeros((n_samples, len(r_grid)))
    
    print(f"开始 {n_samples} 次蒙特卡洛采样计算...")
    for k in range(n_samples):
        # 采样 Delta_E
        Delta_E_k = np.random.normal(Delta_E_mean, Delta_E_std)
        
        # 计算四种方法（对于方法一，结果与 Delta_E 无关，但我们仍将其复制存入）
        F_m1_samples[k, :] = F_m1_single
        
        F_m2_samples[k, :] = calc_method2(r_grid, Delta_E_k)
        F_m3_samples[k, :], n_star_samples[k, :] = calc_method3_continuous(r_grid, Delta_E_k)
        F_m4_samples[k, :] = calc_method4_adiabatic(r_grid, Delta_E_k)
        
    # 计算平均值
    F_m1_mean = np.mean(F_m1_samples, axis=0)
    F_m2_mean = np.mean(F_m2_samples, axis=0)
    F_m3_mean = np.mean(F_m3_samples, axis=0)
    F_m4_mean = np.mean(F_m4_samples, axis=0)
    n_star_mean = np.mean(n_star_samples, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- 子图1：四种方法的平均 F_eff(r) 对比 ---
    ax1.set_yscale('linear')
    ax1.plot(r_m1, F_m1_mean, '-', color='blue', label='"Gibbs" Free Energy', linewidth=lines_linewidth)
    ax1.plot(r_grid, F_m2_mean, '-', color='red', label='Strict free energy(2-state)', linewidth=lines_linewidth)
    ax1.plot(r_grid, F_m3_mean, '-.', color='green', label='Strict free energy', linewidth=lines_linewidth)
    ax1.plot(r_grid, F_m4_mean, '--', color='orange', label='Adiabatic free energy', linewidth=lines_linewidth)
    
    ax1.set_xlabel('End-to-end Length $r$', fontsize=label_fontsize)
    ax1.set_ylabel('$F_{\\text{eff}}(r)$', fontsize=label_fontsize)
    ax1.set_title('Average Effective Free Energy Landscape', fontsize=title_fontsize, pad=20)
    ax1.set_xlim(0, r_max)
    ax1.set_ylim(0, 30.0)
    ax1.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='upper left')
    
    # --- 子图2：n* 的采样与平均 ---
    # 绘制100次采样的灰线
    for k in range(n_samples):
        # 第一个点 r=0 对应的 n* 可能会有波动，但不影响整体
        ax2.plot(r_grid, n_star_samples[k, :], '-', color='gray', alpha=0.6, linewidth=1)
    
    # 绘制平均值红线
    ax2.plot(r_grid, n_star_mean, '-', color='red', linewidth=3, label='Mean $n^*(r)$')
    
    ax2.set_xlabel('$r$', fontsize=label_fontsize)
    ax2.set_ylabel('$n^*(r)$', fontsize=label_fontsize)
    ax2.set_title('Unfolding Fraction under Structural Heterogeneity', fontsize=title_fontsize, pad=20)
    ax2.set_xlim(0, 10.0)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='lower right')
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
        ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)
            
    plt.tight_layout()
    
    # ============ 按路径存储 ============
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'F_eff_comparison_heterogeneity.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")

if __name__ == "__main__":
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Single_chain/results' 
    plot_comparison(output_dir)