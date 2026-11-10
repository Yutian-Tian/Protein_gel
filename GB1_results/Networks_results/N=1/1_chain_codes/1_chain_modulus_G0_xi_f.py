"""
================================================================================
 1-chain 网络初始模量 G0–ξf 可视化（1_chain_modulus_G0_xi_f.py）

 模型计算全部来自 1_chain_common.py，本脚本只负责：
   · 调用 modulus_scan() 在完整参考分布（不截断）下扫描 G0(xi_f)
   · 点线图绘制（folded 圆点 / protein 方块 / Gibbs 菱形）与橡胶模量 G0=3 参考线
   · CSV 与参数 JSON 落盘（只输出 PNG）

 直接运行：python 1_chain_modulus_G0_xi_f.py
 输出位置：同目录 Figures/1_chain_split/modulus/
================================================================================
"""

import os
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
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

# 输出目录：默认在本文件旁的 Figures/1_chain_split/modulus
OUTDIR = Path(__file__).resolve().parent / 'Figures'

# 曲线标记：folded 圆点、protein 方块、Gibbs 菱形
SCAN_MARKERS = {'folded': 'o', 'protein': 'o', 'gibbs': 'o'}
SCAN_COLORS = {'folded': 'blue', 'protein': 'red', 'gibbs': 'black'}


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


def plot_initial_modulus(xis, values):
    """8×6 英寸初始模量点线图：保留橡胶模量 G0=3 黑色虚线参考线。

    values 为 (n_xi, 3) 数组，列序为 [folded, protein, gibbs]。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for kind, j in [('folded', 0), ('protein', 1), ('gibbs', 2)]:
        color = SCAN_COLORS[kind]
        label = {'folded': 'WLC folded', 'protein': 'N=1 protein networks',
                 'gibbs': 'Gibbs network model'}[kind]
        ax.plot(xis, values[:, j], color=color,
                linestyle='-.' if kind == 'gibbs' else '-',
                marker=SCAN_MARKERS[kind], markersize=lines_markersize,
                markerfacecolor=color, linewidth=lines_linewidth, label=label)
    ax.axhline(3, color='gray', linestyle='--', linewidth=lines_linewidth,
            label=r'$G_0^{Rubber} = 3.0$', zorder=1)
    
    ax.set_xlabel(r'Folded length $\xi_f/[l_p]$', fontsize=label_fontsize)
    ax.set_ylabel(r'$G_0/\rho k_BT$', fontsize=label_fontsize)
    ax.set_title('Initial modulus',
                 fontsize=title_fontsize, pad=20)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.08*values.max())
    style_axis(ax)
    ax.legend(fontsize=0.7*legend_fontsize, framealpha=.9, edgecolor='none')
    save_figure(fig, '1_chain_G0_vs_xi_f_N1')


def run():
    """完整参考分布下扫描初始模量 G0(xi_f)，CSV 与 PNG 落盘。"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f'Running modulus scan; output: {OUTDIR}', flush=True)
    started = time.perf_counter()
    xis, values = chain1_common.modulus_scan()
    np.savetxt(OUTDIR/'1_chain_G0_vs_xi_f_N1.csv', np.column_stack((xis, values)),
               delimiter=',', header='xi_f_over_lp,G0_folded,G0_protein,G0_gibbs', comments='')
    plot_initial_modulus(xis, values)
    # 参数与版本信息落盘，保证结果可复现
    metadata = dict(N=chain1_common.N, xi_f_default=chain1_common.XI_F,
                    mu=chain1_common.mu, Delta_E=chain1_common.Delta_E,
                    reference='full (untruncated) reference distribution',
                    xi_min=1., xi_max=100., xi_points=100,
                    modulus_definition='initial nominal uniaxial slope; shear modulus=G0/3',
                    modulus_method='analytic integral <r^2*f_prime+4*r*f>/5; no curve fit',
                    gibbs=dict(K1=chain1_common.K1, K2=chain1_common.K2,
                               reference='full annealed protein reference from 1.png',
                               stiffness="fcp(x)/(Lc(f)+x*Lc_prime(f)*fcp(x))"),
                    elapsed_seconds=time.perf_counter()-started,
                    python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, matplotlib=matplotlib.__version__)
    (OUTDIR/'1_chain_G0_vs_xi_f_N1_parameters.json').write_text(
        json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Completed modulus scan. Results: {OUTDIR}', flush=True)


if __name__ == '__main__':
    run()
