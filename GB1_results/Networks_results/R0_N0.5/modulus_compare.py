"""
绘制不同 R₀ 下的微分模量 G = dσ/dλ 随拉伸比 λ 的变化（对数坐标）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ============ 字体设置 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# 全局绘图参数
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
xi_f = 3.6          # 折叠态持续长度
alpha = 7.6         # 解折叠系数
N = 10.0            # domain 数量
k1 = 6.5
k2 = 1.50
R0_val = [1.5, 5.0, 10.0, 15.0, 20.0]   # 要绘制的初始末端距列表
lambda_max = 300.0  # 最大拉伸比（绘图上限）

def Lc(f):
    """外力 f 下的平均轮廓长度"""
    return N * xi_f * (0.5 * (alpha + 1) + 0.5 * (alpha - 1) * np.tanh(k1 * (f - k2)))

def MSforce(x):
    """Marko-Siggia 单链力-伸长关系"""
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def StressOptimization(R0, N, r_val, f_val):
    """
    计算三链网络模型的应力-拉伸关系
    输入：R0（初始末端距），N（domain数），单链的 r 和 f 数组
    输出：拉伸比 lambda 和 应力 sigma
    """
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

    # 确保 λ=1, σ=0
    if np.abs(lambda_[0] - 1.0) > 1e-12:
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        sigma[0] = 0.0

    return lambda_, sigma

def ModulusMS(R0):
    """计算给定 R0 下的模量曲线（使用 Marko-Siggia 单链模型）"""
    x_MS = np.linspace(0.01, 0.99, 3000)
    f_MS = MSforce(x_MS)
    r_MS = x_MS * Lc(f_MS)
    lambda_, stress_ = StressOptimization(R0, N, r_MS, f_MS)
    modulus_ = np.gradient(stress_, lambda_)
    return lambda_, modulus_

def create_visualization(save_dir=None):
    """绘制并保存模量对比图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    for R0 in R0_val:
        lambda_, modulus = ModulusMS(R0)
        ax.scatter(lambda_, modulus, label=f'$R_0={R0:.1f}$', s=lines_markersize, alpha=0.8)

    ax.set_xlabel('Stretch ratio $\\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Modulus $G / \\rho k_B T$', fontsize=label_fontsize)
    ax.set_title('Modulus $G=\\partial \\sigma / \\partial \\lambda$',
                 fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.8, edgecolor='none', loc=(0.1, 0.65))

    ax.set_xlim(1.01, lambda_max)
    ax.set_ylim(0.001, 1000.0)

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
        path = os.path.join(save_dir, 'modulus_compare.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"模量曲线已保存至: {path}")

def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results'   # 可修改为你希望的输出路径
    create_visualization(save_dir=output_dir)

if __name__ == "__main__":
    main()