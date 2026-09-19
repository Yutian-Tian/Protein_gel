import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid, simpson

# ============ 字体与绘图风格设置 (沿用你的设置) ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 28
label_fontsize = 28
tick_fontsize = 24
legend_fontsize = 20

axes_linewidth = 2
grid_alpha = 0.4
lines_linewidth = 3
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
    'axes.linewidth': axes_linewidth,
    'grid.alpha': grid_alpha,
    'lines.linewidth': lines_linewidth,
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
Delta_E = 11.7      
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

def U_n(n):
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

def F_d(r, n):
    return F_WLC(r, n) + U_n(n)

# ===================== 方法一：唯象力学积分 (可推广至N个domain) =====================
def calc_phenomenological_single_domain():
    """计算单域的唯象自由能"""
    x_vals = np.linspace(0, 0.999, 2000)
    f_vals = fc(x_vals)
    n_vals = 0.5 * (1 + np.tanh(k1 * (f_vals - k2)))
    r_vals = x_vals * Lc_of_n(n_vals)
    F_eff = cumulative_trapezoid(f_vals, r_vals, initial=0)
    return r_vals, F_eff

def calc_phenomenological_chain(N, r_single, F_single):
    """基于单域唯象自由能，计算N个domain串联的链自由能"""
    # 假设链是N个domain均匀串联，总伸长量 R = N * r_single
    R_chain = N * r_single
    F_chain = N * F_single
    return R_chain, F_chain

# ===================== 方法二：严格连续积分 + 蒙特卡洛 =====================
def calc_strict_single_domain_Feff(r_grid):
    """方法三：严格连续积分，获取单域 F_eff(r)"""
    F_eff = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= xi_f * mu * 0.999:
            F_eff[i] = 1e10
            continue
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        res = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 1.0), method='bounded')
        F_min = res.fun
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff[i] = 1e10
            continue
        n_vals = np.linspace(0, 1, 1000)
        integrand = np.exp(-(F_d(r, n_vals) - F_min))
        integral = simpson(integrand, x=n_vals)
        F_eff[i] = F_min - np.log(max(integral, 1e-300))
    return F_eff - F_eff[0]

def calc_mc_chain_Feff(N, M_samples, r_grid, F_eff_r):
    """蒙特卡洛方法计算N个domain串联的链自由能景观"""
    # 单域径向分布 P_eff(r) = 4 * pi * r^2 * exp(-F_eff(r))
    P_eff_r = 4 * np.pi * r_grid**2 * np.exp(-F_eff_r)
    cdf_r = np.cumsum(P_eff_r) * (r_grid[1] - r_grid[0])
    cdf_r = cdf_r / cdf_r[-1]
    
    # 1. 采样单域模长 (M_samples, N)
    random_u = np.random.rand(M_samples, N)
    r_samples = np.interp(random_u, cdf_r, r_grid)
    
    # 2. 采样三维方向 (M_samples, N, 3)
    vecs = np.random.randn(M_samples, N, 3)
    unit_vecs = vecs / np.linalg.norm(vecs, axis=2, keepdims=True)
    
    # 3. 计算链端端矢量 (M_samples, 3)
    R_vecs = np.sum(r_samples[:, :, np.newaxis] * unit_vecs, axis=1)
    R_mags = np.linalg.norm(R_vecs, axis=1)
    
    # 4. 统计直方图并转化为自由能 F(R) = -ln(P(R) / (4*pi*R^2))
    hist_counts, bin_edges = np.histogram(R_mags, bins=150, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 过滤掉概率极低的尾部噪声，避免 log(0)
    valid_idx = hist_counts > 1e-8
    R_valid = bin_centers[valid_idx]
    P_valid = hist_counts[valid_idx]
    
    F_chain_mc = -np.log(P_valid / (4 * np.pi * R_valid**2))
    return R_valid, F_chain_mc

# ===================== 可视化对比 =====================
def plot_N_comparison(output_dir):
    N_list = [5, 20, 50]  # 需要对比的链长
    M_samples = 200000    # 蒙特卡洛采样次数
    colors = ['blue', 'red', 'green']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- 1. 获取单域唯象和严格自由能 ---
    r_phe, F_phe = calc_phenomenological_single_domain()
    
    r_max_single = 0.98 * xi_f * mu
    r_grid = np.linspace(1e-4, r_max_single, 500)
    F_strict = calc_strict_single_domain_Feff(r_grid)
    
    # 绘制单域自由能对比
    ax1.plot(r_phe, F_phe - F_phe[0], '--', color='darkorange', label='Phenomenological (Method 1)')
    ax1.plot(r_grid, F_strict - F_strict[0], '-', color='purple', label='Strict (Method 3)')
    ax1.set_xlabel('End-to-end Length $r$', fontsize=label_fontsize)
    ax1.set_ylabel('$F_{\\text{eff}}(r)$', fontsize=label_fontsize)
    ax1.set_title('Single Domain Free Energy', fontsize=title_fontsize, pad=20)
    ax1.set_xlim(0, r_max_single)
    ax1.set_ylim(0, 30)
    ax1.grid(True, ls="--", alpha=grid_alpha)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none')
    
    # --- 2. 对比不同N下的链自由能 ---
    for i, N in enumerate(N_list):
        # 唯象方法计算链自由能
        R_phe_chain, F_phe_chain = calc_phenomenological_chain(N, r_phe, F_phe)
        # 调整基线对齐
        F_phe_chain_aligned = F_phe_chain - np.min(F_phe_chain)
        
        # 蒙特卡洛方法计算链自由能
        R_mc_chain, F_mc_chain = calc_mc_chain_Feff(N, M_samples, r_grid, F_strict)
        # 调整基线对齐
        F_mc_chain_aligned = F_mc_chain - np.min(F_mc_chain)
        
        # 绘制曲线
        ax2.plot(R_phe_chain, F_phe_chain_aligned, '--', color=colors[i], 
                 label=f'Phenom. $N={N}$', linewidth=2.5, alpha=0.8)
        ax2.plot(R_mc_chain, F_mc_chain_aligned, '-', color=colors[i], 
                 label=f'MC Strict $N={N}$', linewidth=3)
        
    ax2.set_xlabel('Chain End-to-end Length $R$', fontsize=label_fontsize)
    ax2.set_ylabel('$\\Delta F_{\\text{chain}}(R)$', fontsize=label_fontsize)
    ax2.set_title('Free Energy Landscape vs $N$', fontsize=title_fontsize, pad=20)
    ax2.set_xlim(0, max(N_list) * r_max_single)
    ax2.set_ylim(0, 50)
    ax2.grid(True, ls="--", alpha=grid_alpha)
    ax2.legend(fontsize=16, framealpha=0.9, edgecolor='none')
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
        ax.tick_params(axis='y', which='major', direction=ytick_direction, right=ytick_right)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)
            
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'F_chain_N_comparison.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"图表已保存至: {save_path}")


if __name__ == "__main__":
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Single_chain/results' 
    plot_N_comparison(output_dir)