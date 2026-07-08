"""
绘制初始模量 G₀ 随初始末端距 R₀ 的变化，并与理论公式及 R₀² 标度对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ============ 字体设置（与脚本一完全相同） ============
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
legend_fontsize = 20
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

# ============ 物理参数 ============
xi_f = 3.6
alpha = 7.6
N = 10.0
k1 = 6.5
k2 = 1.50
R0_val = [1.5, 5.0, 10.0, 15.0, 20.0]   # 需要计算的 R₀ 点
Rtheo_points = 200                       # 理论曲线采样点数

def Lc(f):
    return N * xi_f * (0.5 * (alpha + 1) + 0.5 * (alpha - 1) * np.tanh(k1 * (f - k2)))

def MSforce(x):
    return np.where(x < 0.99,
                    0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                    np.inf)

def StressOptimization(R0, N, r_val, f_val):
    """与脚本一完全相同的应力计算函数"""
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

def initial_modulus(R0, fit_points=5):
    """
    数值计算初始模量 G₀ = dσ/dλ|_(λ=1)
    利用靠近 λ=1 的若干点线性拟合 σ = G₀ * (λ - 1)
    """
    x_MS = np.linspace(0.01, 0.99, 3000)
    f_MS = MSforce(x_MS)
    r_MS = x_MS * Lc(f_MS)

    lambda_, sigma = StressOptimization(R0, N, r_MS, f_MS)

    lam_fit = lambda_[:fit_points]
    sig_fit = sigma[:fit_points]

    A = (lam_fit - 1.0).reshape(-1, 1)
    G0, _, _, _ = np.linalg.lstsq(A, sig_fit, rcond=None)
    return G0[0]

def calculate_G0(R0):
    """理论公式计算 G₀"""
    L_c = N * xi_f
    x0 = R0 / L_c
    if x0 >= 1:
        raise ValueError(f"x0 = {x0} 必须小于 1，请检查 R₀ 是否小于 L_c={L_c}")
    term1 = x0 / (2 * (1 - x0)**3)
    term2 = 1 / (4 * (1 - x0)**2)
    term3 = 2 * x0
    bracket = term1 + term2 + term3 - 0.25
    coeff = 1.5 * R0
    return coeff * bracket

def plot_G0_vs_R0(R0_val, save_dir=None):
    """绘制 G₀ 随 R₀ 变化图并保存"""
    G0_val = [initial_modulus(r0) for r0 in R0_val]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(R0_val, G0_val, 'o', markerfacecolor='none',
            label='Optimization', markeredgewidth=2, markersize=15, zorder=4)

    # 理论曲线
    R0_theo = np.linspace(0.1, np.float64(R0_val[-1]), Rtheo_points)
    G0_theo = [calculate_G0(r0) for r0 in R0_theo]
    ax.plot(R0_theo, G0_theo, '-', linewidth=lines_linewidth,
            label='Theoretical', alpha=0.8, zorder=5)

    # 参考线 y ∝ R₀²
    ref_x = [R0_theo[0], R0_theo[-1]]
    ref_y = [G0_theo[0], G0_theo[0] * (R0_theo[-1] / R0_theo[0])**2]
    ax.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5,
            label=r'$G_0 \propto R_0^2$', zorder=3)

    ax.set_xlabel('Initial end-to-end distance $R_0$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Initial modulus $G_0$ vs $R_0$  (N={int(N)})',
                 fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    ax.set_xlim(1, np.float64(R0_val[-1]) + 1.0)
    ax.set_ylim(0.1, 150.0)

    ax.tick_params(axis='both', which='major', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width,
                   length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width*0.75,
                   length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'G0_vs_R0.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"G0 vs R0 曲线已保存至: {path}")

def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Multi_chains'   # 可修改为你希望的输出路径
    plot_G0_vs_R0(R0_val, save_dir=output_dir)

if __name__ == "__main__":
    main()