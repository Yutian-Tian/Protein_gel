"""
绘制 WLC 网络初始模量 G0 随链长 L 的变化（3-chain 与 full-chain 分图绘制）
修改点：严格匹配图片公式 p(x;L) = exp[-L*phi(x)] / Z，去除 x^2 因子
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
from scipy.integrate import quad

# ============ 字体设置（保持原样） ============
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
kR = 2.68
xi_f = 3.6

# ===================== 基础物理函数 =====================
def phi(x):
    """无量纲自由能函数"""
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def fc(x):
    """力 f_c(x)"""
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def fcp(x):
    """力对 x 的导数 f_c'(x)"""
    return 0.5 * (1 - x)**(-3) + 1.0

# 图片中 G0 = nL/2 * ∫ x [x f' + f] p(x) dx，展开为 x*(0.5*x*f' + 0.5*f)
# 因此 3-chain 对应 c1=0.5, c2=0.5
C1_3CHAIN = 0.5
C2_3CHAIN = 0.5
C1_FULL = 0.2
C2_FULL = 0.8

# ===================== 图片公式的理论解（3-chain） =====================
def G0_theory_3chain(N, kR=kR, xi_f=xi_f):
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

# ===================== 精确数值积分（严格匹配图片） =====================
def G0_quad(L, c1, c2, n_net=1.0):
    """
    严格按图实现：G0 = nL * ∫ x*(c1*x*f' + c2*f) * p(x) dx
    其中 p(x) = exp(-L*phi(x)) / Z
    """
    def integrand_num(x):
        return x * (c1*x*fcp(x) + c2*fc(x)) * np.exp(-L*phi(x))
    def integrand_den(x):
        return np.exp(-L*phi(x))
    
    # 分段积分提高稳定性
    Z = quad(integrand_den, 0, 1, limit=200, points=[0.99])[0]
    I = quad(integrand_num, 0, 1, limit=200, points=[0.99])[0]
    return n_net * L * I / Z

# ===================== 蒙特卡洛采样（已修正为无x^2的分布） =====================
def metropolis_sample(L, n_samples=100000, burn_in=10000):
    x = 0.05
    sigma = 0.1
    samples = []
    accept_count = 0
    for i in range(burn_in + n_samples):
        x_prop = x + np.random.normal(0, sigma)
        if x_prop < 0 or x_prop >= 1:
            ratio = 0
        else:
            # 【关键修改】去除 x^2 因子，完全对应图片中的 p(x) = exp(-L*phi(x))
            log_ratio = - L * (phi(x_prop) - phi(x))
            ratio = np.exp(log_ratio) if log_ratio < 0 else 1.0
        if np.random.rand() < ratio:
            x = x_prop
            accept_count += 1
        if i >= burn_in:
            samples.append(x)
        if i % 1000 == 0 and i > 0:
            accept_rate = accept_count / i
            if accept_rate < 0.25:
                sigma *= 0.7
            elif accept_rate > 0.6:
                sigma *= 1.3
    return np.array(samples)

def G0_mc(L, c1, c2, n_net=1.0, n_samples=50000, burn_in=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_samples = metropolis_sample(L, n_samples, burn_in)
    values = x_samples * (c1 * x_samples * fcp(x_samples) + c2 * fc(x_samples))
    avg = np.mean(values)
    return n_net * L * avg, np.std(values) / np.sqrt(len(values))

# ===================== 重要性采样（已修正） =====================
def G0_importance(L, c1, c2, n_net=1.0, n_samples=200000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sigma_x = np.sqrt(2 / (3 * L))
    x_samples = np.random.normal(0, sigma_x, n_samples)
    x_samples = x_samples[(x_samples >= 0) & (x_samples < 1)]
    if len(x_samples) < 1000:
        return G0_mc(L, c1, c2, n_net, n_samples=5000, burn_in=500)
    
    log_q = -0.5 * (x_samples / sigma_x)**2
    # 【关键修改】同样去除 x^2 因子
    log_p = - L * phi(x_samples)
    log_w = log_p - log_q
    w = np.exp(log_w - np.max(log_w))
    w /= np.sum(w)
    
    values = x_samples * (c1 * x_samples * fcp(x_samples) + c2 * fc(x_samples))
    avg = np.sum(w * values)
    return n_net * L * avg, 0.0

# ===================== 分布检验绘图函数（已修正） =====================
def plot_distribution_check(L, n_samples=50000, burn_in=5000, save_dir=None):
    samples = metropolis_sample(L, n_samples=n_samples, burn_in=burn_in)

    # 严格匹配图片，归一化常数 Z 也不包含 x^2
    Z = quad(lambda x: np.exp(-L * phi(x)), 0, 1, limit=200, points=[0.99])[0]

    def pdf_theory(x):
        return np.exp(-L * phi(x)) / Z

    def cdf_theory(x):
        return quad(lambda t: np.exp(-L * phi(t)) / Z, 0, x, limit=200)[0]

    cdf_vec = np.vectorize(cdf_theory)

    x_grid = np.linspace(0.001, 0.999, 500)
    pdf_vals = pdf_theory(x_grid)
    cdf_vals = cdf_vec(x_grid)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.hist(samples, bins=50, density=True, alpha=0.5, edgecolor='black')
    ax1.plot(x_grid, pdf_vals, 'r-', linewidth=5)
    ax1.set_xlabel('$x$', fontsize=label_fontsize)
    ax1.set_ylabel('Probability Density', fontsize=label_fontsize, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    sorted_samples = np.sort(samples)
    y_ecdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    ax2.step(sorted_samples, y_ecdf, where='post', color='blue', linewidth=2, label='Empirical CDF')
    ax2.plot(x_grid, cdf_vals, 'g--', linewidth=5, label='Theoretical CDF')
    ax2.set_ylabel('Cumulative Probability', fontsize=label_fontsize, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_title(f'Distribution check at $L={L}$', fontsize=title_fontsize, pad=20)
    ax1.grid(True, which='major', ls='--', alpha=grid_alpha)
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='none', fontsize=legend_fontsize*0.7)

    # 美化刻度
    ax1.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax1.tick_params(axis='y', which='major', length=6, direction=ytick_direction, right=False)
    ax2.tick_params(axis='y', which='major', length=6, direction=ytick_direction, right=True)

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 3.0)
    ax2.set_ylim(0.0, 1.05)

    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'distribution_check_L{L}_modified.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"分布检验图已保存至: {path}")

# ===================== 绘图函数（保持原样，仅修正内部调用） =====================
def plot_G0_3chain(L_vals, save_dir=None):
    G0_quad_vals = []
    G0_mc_vals = []
    G0_theory_vals = []

    for L in L_vals:
        g0_q = G0_quad(L, C1_3CHAIN, C2_3CHAIN)
        G0_quad_vals.append(g0_q)
        
        if L <= 10:
            g0_m, _ = G0_mc(L, C1_3CHAIN, C2_3CHAIN, n_samples=20000, burn_in=2000)
        else:
            g0_m, _ = G0_importance(L, C1_3CHAIN, C2_3CHAIN, n_samples=50000)
        G0_mc_vals.append(g0_m)
        
        N = L / xi_f
        G0_theory_vals.append(G0_theory_3chain(N))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5e-1, 1e3)
    ax.set_ylim(1e0, 1e2)
    
    ax.plot(L_vals, G0_quad_vals, 'o', markerfacecolor='none', label='Quadrature',
            markeredgewidth=2, markersize=10, color='blue', zorder=4)
    ax.plot(L_vals, G0_mc_vals, '--', color='purple', linewidth=3, label='Monte Carlo', zorder=3)
    ax.plot(L_vals, G0_theory_vals, '-', color='black', linewidth=3, label=f'$k_R={kR:.2f}$', alpha=0.8, zorder=5)
    ax.axvline(x=3.6, color='red', linestyle='--', linewidth=3, label='$L=3.6$')
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=3, label='Rubber Modulus')
    
    ax.set_xlabel('Contour length $L$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / n k_B T$', fontsize=label_fontsize)
    ax.set_title('3-chain topology', fontsize=title_fontsize, pad=20)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'G0_3chain_vs_L_wlc_modified.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"3-chain图已保存至: {path}")

def plot_G0_fullchain(L_vals, save_dir=None):
    G0_quad_vals = []
    G0_mc_vals = []

    for L in L_vals:
        g0_q = G0_quad(L, C1_FULL, C2_FULL)
        G0_quad_vals.append(g0_q)
        
        if L <= 10:
            g0_m, _ = G0_mc(L, C1_FULL, C2_FULL, n_samples=20000, burn_in=2000)
        else:
            g0_m, _ = G0_importance(L, C1_FULL, C2_FULL, n_samples=50000)
        G0_mc_vals.append(g0_m)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5e-1, 1e3)
    ax.set_ylim(1e0, 1e2)
    
    ax.plot(L_vals, G0_quad_vals, 'o', markerfacecolor='none', label='Quadrature',
            markeredgewidth=2, markersize=15, color='green', zorder=4)
    ax.plot(L_vals, G0_mc_vals, '--', color='purple', linewidth=3, label='Monte Carlo', zorder=3)
    ax.axvline(x=3.6, color='red', linestyle='--', linewidth=3, label='$L=3.6$')
    ax.axhline(y=3.0, color='red', linestyle=':', linewidth=3, label='Rubber Modulus')
    
    ax.set_xlabel('Contour length $L$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / n k_B T$', fontsize=label_fontsize)
    ax.set_title('Full-chain topology', fontsize=title_fontsize, pad=20)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'G0_fullchain_vs_L_wlc_modified.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"full-chain图已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    L_vals = np.logspace(-0.5, 3, 100)
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/Mento-Carlo_simulation'
    
    plot_G0_3chain(L_vals, save_dir=output_dir)
    plot_G0_fullchain(L_vals, save_dir=output_dir)
    plot_distribution_check(L=3.6, n_samples=50000, burn_in=5000, save_dir=output_dir)

if __name__ == "__main__":
    main()