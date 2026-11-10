"""
================================================================================
 1-chain 网络参考态分布可视化（1_chain_reference_distribution.py）

 模型计算全部来自 1_chain_common.py，本脚本只负责：
   · 调用 reference_distribution_data() 得到三种网络的完整参考态 p0(r)
   · 分布函数图绘制（纯线型，不用点线标记）、CSV 落盘

 直接运行：python 1_chain_reference_distribution.py
 输出位置：同目录 Figures/1_chain_split/reference/
================================================================================
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'chain1_common', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '1_chain_common.py'))
chain1_common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(chain1_common)

# ============ 字体与绘图风格设置（命名常量集中管理，便于全局统一调整） ============
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
lines_linewidth = 3
lines_markersize = 5

figure_dpi = 100
savefig_dpi = 300
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_left = True
ytick_right = True

if os.path.exists(font_path):
    # 仅 Linux 存在该字体文件；Windows 上自动回退到系统 serif 字体
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
    'pdf.fonttype': 42,   # 字体嵌入设置（对 PNG 无影响，保留以兼容旧习惯）
    'ps.fonttype': 42,
})

# 输出目录：默认在本文件旁的 Figures/1_chain_split/reference
OUTDIR = Path(__file__).resolve().parent / 'Figures'


def style_axis(ax):
    """按统一模板设置坐标轴主次刻度方向/线宽与边框线宽。"""
    ax.grid(True, which='major', linestyle='--', alpha=grid_alpha)
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


def save_figure(fig, stem):
    """以 300 dpi PNG 保存并关闭图。"""
    fig.tight_layout()
    fig.savefig(OUTDIR/f'{stem}.png', dpi=savefig_dpi,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_reference_distribution(r, protein, rf, folded):
    """三种网络的完整参考态分布：光滑密度曲线用纯线型。

    解折叠 WLC 继承蛋白参考态（绿点划线与红线重合），
    折叠 WLC 采用自身 WLC 平衡分布（蓝虚线）。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, protein, 'r-', label=r'$N=1$ protein network')
    ax.plot(rf, folded, 'b--', label='WLC folded')
    ax.plot(r, protein, 'g-.', label='WLC unfolded')
    ax.set_xlabel(r'End-to-end distance $r\,/\,l_p$', fontsize=label_fontsize)
    ax.set_ylabel(r'$p_0(r)$', fontsize=label_fontsize)
    ax.set_title('Reference state distribution', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, 10)
    style_axis(ax)
    ax.legend(fontsize=0.7*legend_fontsize, framealpha=.9, edgecolor='none',
              loc='best')
    save_figure(fig, 'Reference_distribution')


def run():
    """三种网络完整参考态分布：CSV + PNG。"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f'Running reference distribution; output: {OUTDIR}', flush=True)
    xi = chain1_common.XI_F
    r, protein, rf, folded = chain1_common.reference_distribution_data(xi=xi)
    np.savetxt(OUTDIR/'Reference_distribution.csv',
               np.column_stack((r, protein, protein)), delimiter=',',
               header='r_over_lp,p0_protein_full,p0_WLC_unfolded_full', comments='')
    np.savetxt(OUTDIR/'Reference_distribution_folded.csv',
               np.column_stack((rf, folded)), delimiter=',',
               header='r_over_lp,p0_WLC_folded_full', comments='')
    plot_reference_distribution(r, protein, rf, folded)
    print(f'Completed reference distribution. Results: {OUTDIR}', flush=True)


if __name__ == '__main__':
    run()
