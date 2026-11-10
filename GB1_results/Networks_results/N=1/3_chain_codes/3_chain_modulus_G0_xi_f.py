"""
================================================================================
 3-chain 网络初始模量 G0–ξf 可视化（3_chain_modulus_G0_xi_f.py）

 模型计算全部来自 3_chain_common.py，本脚本只负责：
   · 扫描 ξ_f，计算退火 / WLC / Gibbs 三方案的初始模量 G0
   · 调用 validate_modulus() + validate_gibbs_modulus() 做数值验证
   · 点线图绘制、CSV/验证文本落盘（只输出 PNG）

 直接运行：python 3_chain_modulus_G0_xi_f.py
 输出位置：同目录 Figures/G0_vs_xi_f_N*.png
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
lines_linewidth = 3
lines_markersize = 7

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


def plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=None):
    """扫描 ξ_f 计算三方案初始模量，并绘制 G0–ξ_f 点线图。"""
    G0_quad_vals = []
    G0_WLC_vals = []
    G0_gibbs_vals = []
    frozen_vals, correction_vals = [], []

    print(f"正在计算 N={N_single} 下的数据 (退火有效自由能)...")
    for xi_f in xi_f_vals:
        pre = chain3_common.build_annealed(N_single, xi_f)
        g, frozen, correction = chain3_common.modulus_components(pre)
        G0_gibbs_vals.append(chain3_common.G0_3chain_gibbs(pre))
        G0_quad_vals.append(g)
        frozen_vals.append(frozen)
        correction_vals.append(correction)
        G0_WLC_vals.append(chain3_common.G0_3chain_WLC(N_single, xi_f))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(min(xi_f_vals), max(xi_f_vals))
    ax.set_ylim(0, 1.08 * max(max(G0_quad_vals), max(G0_WLC_vals),
                              max(G0_gibbs_vals), 3))

    # ---- marker 抽稀与统一尺寸 ----
    # 100 个数据点下每 ~5 点显示一个 marker，避免重叠。
    markevery = max(1, len(xi_f_vals) // 20)
    marker_kw = dict(markersize=1.6*lines_markersize, markeredgewidth=1.0,
                     markevery=markevery)

    # 红线：退火有效自由能的模量，含条件力方差修正
    ax.plot(xi_f_vals, G0_quad_vals, '-s', color='red',
            linewidth=lines_linewidth,
            label=f'$N={N_single}$ protein network',
            zorder=6, **marker_kw)

    # 蓝线：固定轮廓长度 WLC
    ax.plot(xi_f_vals, G0_WLC_vals, '-o', color='blue',
            linewidth=lines_linewidth,
            label='WLC network',
            zorder=5, **marker_kw)

    # 黑线：Gibbs network model
    ax.plot(xi_f_vals, G0_gibbs_vals, '-^', color='black',
            linewidth=lines_linewidth,
            label='Gibbs network model',
            zorder=7, **marker_kw)

    ax.axhline(y=3.0, color='gray', linestyle='--',
               linewidth=lines_linewidth,
               label='$G_0^{Rubber}=$3.0', zorder=1)

    ax.set_xlabel('Folded length $\\xi_\\mathrm{f}$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / \\rho k_\\mathrm{B} T$', fontsize=label_fontsize)
    ax.set_title('Initial Modulus', fontsize=title_fontsize, pad=20)

    style_axis(ax)

    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f'G0_vs_xi_f_N{N_single}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        np.savetxt(os.path.join(save_dir, f'G0_vs_xi_f_N{N_single}.csv'),
                   np.column_stack((xi_f_vals, G0_quad_vals, G0_WLC_vals,
                                    frozen_vals, correction_vals,
                                    G0_gibbs_vals)), delimiter=',',
                   header='xi_f_over_lp,G0_annealed_over_rho_kBT,'
                          'G0_WLC_over_rho_kBT,G0_frozen_over_rho_kBT,'
                          'variance_correction_over_rho_kBT,'
                          'G0_gibbs_over_rho_kBT',
                   comments='')
        print(f"图表已保存至: {path}")
    plt.close(fig)


# ===================== 主程序入口 =====================
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    N_single = 1
    xi_f_vals = np.linspace(1.0, 100.0, 100)
    validation = (chain3_common.validate_modulus() +
                  chain3_common.validate_gibbs_modulus())
    with open(os.path.join(output_dir, 'Modulus_N1_validation.txt'), 'w') as report:
        report.write(f'N=1, mu={chain3_common.mu}, Delta_E={chain3_common.Delta_E}, '
                     f'k1={chain3_common.K1}, k2={chain3_common.K2}, '
                     f'kBT=lp=1; full equilibrium reference\n')
        report.write('\n'.join(validation) + '\n')

    plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=output_dir)


if __name__ == "__main__":
    main()
