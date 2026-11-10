"""
================================================================================
 3-chain 网络参考态分布可视化（3_chain_reference_distribution.py）

 模型计算全部来自 3_chain_common.py，本脚本只负责绘制两类参考态分布：
   · 径向参考分布 p0(r)：protein 网络变形前（λ=1）的链端距分布
     （均匀网格生成，含 WTOL 尾部截断，与有限参考系综本构共用同一分布）
   · 内部态边缘分布 p(n)：对端距完整积分后的解折叠状态分布
     （完整平衡系综，模量计算所用的同一参考态）

 分布函数图按约定保持纯线型（不用点线标记），其中 p(n) 用对数纵轴。

 直接运行：python 3_chain_reference_distribution.py
 输出位置：同目录 Figures/Reference_distribution_p0r_N*.png、
            Figures/pn_distribution_N*_xi*.png
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'chain3_common', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '3_chain_common.py'))
chain3_common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(chain3_common)

# ============ 字体与绘图风格设置（与 3_chain_Constitutive_curve_N1.py 对齐） ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 30
tick_fontsize = 35
legend_fontsize = 30
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
xtick_minor_width = 1
ytick_minor_width = 1
xtick_minor_size = 5
ytick_minor_size = 5
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
    'xtick.minor.width': xtick_minor_width,
    'ytick.minor.width': ytick_minor_width,
    'xtick.minor.size': xtick_minor_size,
    'ytick.minor.size': ytick_minor_size,
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


def style_axis(ax):
    """统一坐标轴样式：网格、主次刻度朝内、边框线宽。"""
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.tick_params(axis='x', which='major', length=6,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width,
                   direction=ytick_direction, right=ytick_right)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, width=1,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=4, width=1,
                   direction=ytick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)


def plot_reference_p0r(N=1, save_dir=None):
    """绘制 protein 网络的径向参考分布 p0(r)（截断后的有限参考系综）。"""
    r_full, u_full = chain3_common.single_domain_u()
    ch = chain3_common.ChainN1(r_full, u_full)
    print(f'[p0(r)] 支撑区间 [0, {ch.r[-1]:.3f}] lp，'
          f'有效网格点数 {len(ch.r)}，R0 = {ch.R0:.4f} lp')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    # 原始平衡权重 r²·u（截断前，画到完整支撑）与截断后固定的 p0 对比，
    # 截断位置以外的尾部差异清晰可见。
    p0_raw = r_full**2 * u_full
    p0_raw /= chain3_common.trapz_func(p0_raw, r_full)
    ax.plot(r_full, p0_raw, '--', color='gray', linewidth=0.6*lines_linewidth,
            label='Full equilibrium')
    ax.plot(ch.r, ch.p0, '-', color='red', linewidth=lines_linewidth,
            label=rf'$N={N}$ protein network')

    ax.set_xlabel(r'End-to-end distance $r / l_\mathrm{p}$',
                  fontsize=label_fontsize)
    ax.set_ylabel(r'Reference distribution $p_0(r)$',
                  fontsize=label_fontsize)
    ax.set_title('Reference state', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, 1.5 * ch.r[-1])
    ax.set_ylim(bottom=0)

    style_axis(ax)
    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'Reference_distribution_p0r_N{N}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"p0(r) 分布图已保存至: {path}")
    plt.close(fig)


def plot_pn_vs_n(N_single, xi_f, save_dir=None):
    """绘制内部态边缘分布 p(n)（对数纵轴，完整平衡系综）。"""
    n_grid, p_n = chain3_common.calculate_pn(N_single, xi_f)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('log')

    # 分布函数图：纯线型 + 空心散点强调峰位
    ax.plot(n_grid, p_n, '-', color='purple', linewidth=lines_linewidth,
            label=f'$\\xi_\\mathrm{{f}}={xi_f}$')
    ax.plot(n_grid, p_n, 'o', color='purple', markersize=6,
            markerfacecolor='white')

    ax.set_xlabel('State variable $n$', fontsize=label_fontsize)
    ax.set_ylabel('Probability density $p(n)$', fontsize=label_fontsize)
    ax.set_title(f'Marginal Distribution ($N={N_single}$)',
                 fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, 1)

    style_axis(ax)
    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir,
                            f'pn_distribution_N{N_single}_xi{xi_f}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"p(n) 分布图已保存至: {path}")
    plt.close(fig)


# ===================== 主程序入口 =====================
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    plot_reference_p0r(N=1, save_dir=output_dir)
    # plot_pn_vs_n(N_single=1, xi_f=chain3_common.xi_f, save_dir=output_dir)


if __name__ == "__main__":
    main()
