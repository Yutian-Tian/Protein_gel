"""
两态蛋白域解折叠模型：3-chain 网络 + 整链仿射
单位约定: k_B T = l_p = rho = 1
能垒Delta_E = 11.7, 展开长度系数 mu = 7.6, 折叠态轮廓长度 xi_f = 3.6
计算剪切模量 G0(N), 杨氏模量 E0(N), 以及不可压单轴拉伸本构曲线 sigma(lambda)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ============================================================
# 字体与绘图风格设置
# ============================================================
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

# ============================================================
# 物理参数（单位: kT = lp = rho = 1）
# ============================================================
xi_f = 3.6                 # 折叠态轮廓长度
mu   = 7.6                 # 展开长度比例系数
xi_u = mu * xi_f           # 展开态轮廓长度
dE   = 11.7                # 展开能垒

# ============================================================
# 基础物理函数
# ============================================================
def F_WLC(r, Lc):
    """Marko-Siggia 自由能（无量纲，kT=1, lp=1）"""
    x = np.clip(r / Lc, 0.0, 1.0 - 1e-12)
    return (Lc / 4.0) * x**2 * (3.0 - 2.0 * x) / (1.0 - x)

def compute_pd(r_grid):
    """两态单域径向分布 p_d(r)"""
    p_fold   = 4.0 * np.pi * r_grid**2 * np.exp(-F_WLC(r_grid, xi_f))
    p_unfold = 4.0 * np.pi * r_grid**2 * np.exp(-F_WLC(r_grid, xi_u) - dE)
    p_d = p_fold + p_unfold
    Z = trapezoid(p_d, r_grid)
    return p_d / Z

def compute_ztilde(r_grid, p_d, k_grid):
    """单域特征函数 \tilde z(k)"""
    sinc_mat = np.sinc(k_grid[:, None] * r_grid[None, :] / np.pi)
    return trapezoid(p_d[None, :] * sinc_mat, r_grid, axis=1)

def compute_chain(N, k_grid, z_k):
    """整链分布 p_ch(R) 与自由能 F_ch(R)"""
    R_max = N * xi_u * 1.2
    n_R   = max(1200, min(4000, N * 200))
    R_grid = np.linspace(1e-6, R_max, n_R)

    ln_z = np.log(np.maximum(z_k, 1e-300))
    N_ln_z = N * ln_z

    integrand = (k_grid[:, None] * np.sin(k_grid[:, None] * R_grid[None, :])
                 * np.exp(N_ln_z[:, None]))
    I_R = trapezoid(integrand, k_grid, axis=0)

    p_ch = I_R / (2.0 * np.pi**2 * R_grid)
    p_ch = np.maximum(p_ch, 1e-300)

    p_ch_r = 4.0 * np.pi * R_grid**2 * p_ch
    norm = trapezoid(p_ch_r, R_grid)
    p_ch_r /= norm
    p_ch   /= norm

    F_ch = -np.log(p_ch)     # kT = 1
    return R_grid, p_ch_r, F_ch

def compute_G0(N, k_grid, z_k):
    """初始剪切模量 G0(N) 与杨氏模量 E0(N)"""
    R_grid, p_ch_r, F_ch = compute_chain(N, k_grid, z_k)

    n_win = min(51, (len(R_grid) // 6) * 2 + 1)
    if n_win < 5:
        n_win = 5
    F_s = savgol_filter(F_ch, window_length=n_win, polyorder=3)

    f_ch  = np.gradient(F_s, R_grid)
    F_pp  = np.gradient(f_ch, R_grid)

    avg = trapezoid(p_ch_r * F_pp * R_grid**2, R_grid)

    G0 = (1.0 / 6.0) * (3.0 + avg)     # rho = 1
    E0 = 3.0 * G0
    return G0, E0, R_grid, p_ch_r, F_s, f_ch

def compute_sigma(lam, R_grid, p_ch_r, f_ch, theta_grid, phi_grid):
    """不可压单轴拉伸 Cauchy 应力 σ(λ)"""
    f_interp = interp1d(R_grid, f_ch, kind='cubic',
                        bounds_error=False, fill_value=0.0)

    T, P = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    sin_T = np.sin(T)
    cos_T = np.cos(T)
    cos_P = np.cos(P)
    sin_P = np.sin(P)

    Lambda = np.sqrt(
        lam**2       * sin_T**2 * cos_P**2
        + lam**(-1)  * sin_T**2 * sin_P**2
        + lam**(-1)  * cos_T**2
    )

    dLambda_dlam = (
        lam * sin_T**2 * cos_P**2
        - 0.5 * lam**(-2) * (sin_T**2 * sin_P**2 + cos_T**2)
    ) / Lambda

    R_def = R_grid[:, None, None] * Lambda[None, :, :]
    f_val = f_interp(R_def)

    integrand = (f_val * R_grid[:, None, None]
                 * dLambda_dlam[None, :, :] * sin_T[None, :, :])

    int_phi   = trapezoid(integrand, phi_grid, axis=2)
    int_theta = trapezoid(int_phi, theta_grid, axis=1)

    sigma = lam * trapezoid(p_ch_r * int_theta / (4.0 * np.pi), R_grid)
    return sigma

# ============================================================
# 数据计算
# ============================================================
def compute_all_data(N_values, lam_grid, N_list_sigma,
                     r_grid=None, k_grid=None,
                     theta_grid=None, phi_grid=None):
    """统一计算 G0(N), E0(N), sigma(lambda)"""
    if r_grid is None:
        r_grid = np.linspace(1e-6, xi_u * (1 - 1e-10), 3000)
    if k_grid is None:
        k_grid = np.linspace(1e-6, 40.0 / xi_f, 4000)
    if theta_grid is None:
        theta_grid = np.linspace(0.0, np.pi, 40)
    if phi_grid is None:
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 40)

    p_d = compute_pd(r_grid)
    z_k = compute_ztilde(r_grid, p_d, k_grid)

    # G0(N), E0(N)
    G0_list, E0_list = [], []
    print("计算 G0(N) 与 E0(N) ...")
    for N in N_values:
        G0, E0, _, _, _, _ = compute_G0(N, k_grid, z_k)
        G0_list.append(G0)
        E0_list.append(E0)
        print(f"N = {N:4d},  G0 = {G0:.4f},  E0 = {E0:.4f}")

    # sigma(lambda)
    print("\n计算 sigma(lambda) ...")
    sigma_data = {}
    for N in N_list_sigma:
        _, _, R_grid, p_ch_r, _, f_ch = compute_G0(N, k_grid, z_k)
        sigma_vals = [compute_sigma(lam, R_grid, p_ch_r, f_ch,
                                    theta_grid, phi_grid)
                      for lam in lam_grid]
        sigma_data[N] = np.asarray(sigma_vals)
        print(f"完成 N = {N}")

    return {
        'N':         np.asarray(N_values),
        'G0':        np.asarray(G0_list),
        'E0':        np.asarray(E0_list),
        'lam':       np.asarray(lam_grid),
        'sigma_lam': sigma_data,
    }

# ============================================================
# 通用坐标轴样式
# ============================================================
def _style_log_axes(ax):
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.minorticks_on()

    ax.tick_params(axis='x', which='major', length=6,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='x', which='minor', length=4,
                   width=xtick_major_width, direction=xtick_direction,
                   top=xtick_top)
    ax.tick_params(axis='y', which='major',
                   width=ytick_major_width, direction=ytick_direction,
                   right=ytick_right)
    ax.tick_params(axis='y', which='minor', length=4,
                   width=ytick_major_width, direction=ytick_direction,
                   right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

def _style_linear_axes(ax):
    ax.minorticks_on()
    ax.tick_params(axis='x', which='major', length=6,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='x', which='minor', length=4,
                   width=xtick_major_width, direction=xtick_direction,
                   top=xtick_top)
    ax.tick_params(axis='y', which='major',
                   width=ytick_major_width, direction=ytick_direction,
                   right=ytick_right)
    ax.tick_params(axis='y', which='minor', length=4,
                   width=ytick_major_width, direction=ytick_direction,
                   right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

# ============================================================
# 绘图函数 1: G0(N)
# ============================================================
def plot_G0_vs_N(data, output_dir=None):
    """绘制剪切模量 G0(N)"""
    N_arr  = data['N']
    G0_arr = data['G0']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(N_arr, G0_arr, 'o-', color='blue', linewidth=3,
            markersize=10, markerfacecolor='none', markeredgewidth=2,
            label=r'$G_0$ (WLC + 2-state)')

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=3,
               label=r'Neo-Hookean limit $G_0=\rho k_B T$')

    N_line = np.logspace(np.log10(N_arr[0]), np.log10(N_arr[-1]), 100)
    G_line = G0_arr[0] * (N_line / N_arr[0]) ** (-1)
    ax.plot(N_line, G_line, '--', color='gray', linewidth=2,
            label=r'$G_0 \sim N^{-1}$')

    ax.set_xlabel(r'Number of domains $N$', fontsize=label_fontsize)
    ax.set_ylabel(r'$G_0 / \rho k_B T$', fontsize=label_fontsize)
    ax.set_title('Shear modulus vs. domain number',
                 fontsize=title_fontsize, pad=20)
    ax.grid(True, which='major', ls='--', alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize * 0.8, framealpha=0.9,
              edgecolor='none', loc='best')
    _style_log_axes(ax)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'G0_vs_N_2state.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"G0(N) 图已保存至: {path}")

    return fig

# ============================================================
# 绘图函数 2: E0(N) 与 G0(N)
# ============================================================
def plot_E_vs_N(data, output_dir=None):
    """绘制杨氏模量 E0(N) 与剪切模量 G0(N)"""
    N_arr  = data['N']
    G0_arr = data['G0']
    E0_arr = data['E0']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(N_arr, E0_arr, 'o-', color='crimson', linewidth=3,
            markersize=10, markerfacecolor='none', markeredgewidth=2,
            label=r'$E_0$ (WLC + 2-state)')
    ax.plot(N_arr, G0_arr, 's--', color='steelblue', linewidth=3,
            markersize=10, markerfacecolor='none', markeredgewidth=2,
            label=r'$G_0$ (WLC + 2-state)')

    ax.axhline(y=3.0, color='black', linestyle=':', linewidth=3,
               label=r'Neo-Hookean limit $E_0=3\rho k_B T$')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=3,
               label=r'Neo-Hookean limit $G_0=\rho k_B T$')

    N_line = np.logspace(np.log10(N_arr[0]), np.log10(N_arr[-1]), 100)
    E_line = E0_arr[0] * (N_line / N_arr[0]) ** (-1)
    ax.plot(N_line, E_line, '--', color='lightgray', linewidth=2,
            label=r'$E_0 \sim N^{-1}$')

    ax.set_xlabel(r'Number of domains $N$', fontsize=label_fontsize)
    ax.set_ylabel(r'$E_0 / \rho k_B T$', fontsize=label_fontsize)
    ax.set_title("Young's modulus vs. domain number",
                 fontsize=title_fontsize, pad=20)
    ax.grid(True, which='major', ls='--', alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize * 0.8, framealpha=0.9,
              edgecolor='none', loc='best')
    _style_log_axes(ax)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'E_vs_N_2state.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"E(N) 图已保存至: {path}")

    return fig

# ============================================================
# 绘图函数 3: sigma(lambda)
# ============================================================
def plot_sigma_lambda(data, N_list=None, colors=None,
                      output_dir=None):
    """绘制本构曲线 σ(λ)"""
    lam_grid  = data['lam']
    sigma_lam = data['sigma_lam']

    if N_list is None:
        N_list = sorted(sigma_lam.keys())
    if colors is None:
        base = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        colors = (base * (len(N_list) // len(base) + 1))[:len(N_list)]

    fig, ax = plt.subplots(figsize=(10, 8))

    for N, color in zip(N_list, colors):
        if N not in sigma_lam:
            continue
        ax.plot(lam_grid, sigma_lam[N], '-', color=color, linewidth=3,
                label=rf'$N = {N}$')

    ax.set_xlabel(r'Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'Cauchy stress $\sigma / \rho k_B T$',
                  fontsize=label_fontsize)
    ax.set_title('Constitutive curves (3-chain, 2-state)',
                 fontsize=title_fontsize, pad=20)
    ax.grid(True, which='major', ls='--', alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize * 0.8, framealpha=0.9,
              edgecolor='none', loc='best')
    _style_linear_axes(ax)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'sigma_lambda_2state.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"sigma(lambda) 图已保存至: {path}")

    return fig

# ============================================================
# 主程序
# ============================================================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/relaxed_gel/figures'

    # N 采样与本构曲线 λ 网格
    N_values     = np.linspace(1, 20, 10, dtype=int)
    N_list_sigma = [1, 2, 4, 6, 8, 10]
    lam_grid     = np.linspace(1.0, 10.0, 200)

    # 统一计算
    data = compute_all_data(N_values, lam_grid, N_list_sigma)

    # 每张图独立调用
    plot_G0_vs_N(data, output_dir=output_dir)
    plot_E_vs_N(data, output_dir=output_dir)
    plot_sigma_lambda(data, N_list=N_list_sigma,
                      colors=['blue', 'green', 'red', 'purple'],
                      output_dir=output_dir)


if __name__ == "__main__":
    main()