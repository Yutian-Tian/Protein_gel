"""
================================================================================
 1-chain 网络本构曲线可视化（1_chain_constitutive_curves.py）

 模型计算全部来自 1_chain_common.py，本脚本只负责：
   · 调用 compute_constitutive(kinds=KINDS_GIBBS) 得到四种网络应力曲线
     （folded / protein / unfolded / gibbs tanh）
   · 数值验证文本与参数 JSON 落盘
   · 点线图绘制、CSV/验证文本/参数 JSON 落盘（只输出 PNG）

 直接运行：python 1_chain_constitutive_curves.py
 输出位置：同目录 Figures/
================================================================================
"""

import os
import json
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

# ============ 字体与绘图风格设置（与 3_chain_constitutive_curves.py 对齐） ============
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

# 点线图参数：marker 抽稀间隔与统一尺寸
MARKERSIZE = lines_markersize

# 本构点线图标签与颜色（marker 形状在绘图处内联，与代码2一致）
LABELS = {'folded': 'WLC folded', 'protein': 'Annealed protein network',
          'unfolded': 'WLC unfolded', 'gibbs': 'Gibbs tanh network'}
COLORS = {'folded': 'blue', 'protein': 'red', 'unfolded': 'green',
          'gibbs': 'black'}


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


def plot_constitutive_curve(outdir, N=1):
    """计算并绘制蛋白、两种 WLC 基线及 1.png 定义的 Gibbs tanh 曲线。

    图中的三链能量在本文件夹按既有的一链连续取向平均实现。
    """
    # ---------- 模型计算（1_chain_common.py） ----------
    print(f'=== N = {N} 四模型本构（含 Gibbs tanh） ===')
    xi = chain1_common.XI_F
    res = chain1_common.compute_constitutive(xi=xi,
                                             kinds=chain1_common.KINDS_GIBBS)
    curves = res['curves']

    lam_p, sig_p = curves['protein']
    lam_f, sig_f = curves['folded']
    lam_u, sig_u = curves['unfolded']
    lam_g, sig_g = curves['gibbs']

    print(f'lambda_end protein = {lam_p[-1]:.10g}, sigma_end = {sig_p[-1]:.10g}')
    print(f'lambda_end folded  = {lam_f[-1]:.10g}, sigma_end = {sig_f[-1]:.10g}')
    print(f'lambda_end unfolded= {lam_u[-1]:.10g}, sigma_end = {sig_u[-1]:.10g}')
    print(f'lambda_end gibbs   = {lam_g[-1]:.10g}, sigma_end = {sig_g[-1]:.10g}')

    # ---------- 验证文本落盘 ----------
    validation = res['validation']
    for line in validation:
        print(line)
    with open(os.path.join(outdir, f'1_chain_N{N}_validation.txt'),
              'w', encoding='utf-8') as report:
        report.write('\n'.join(validation) + '\n')

    # ---------- CSV 落盘 ----------
    for kind in LABELS:
        lam, sig = curves[kind]
        np.savetxt(os.path.join(outdir,
                                f'1_chain_constitutive_N{N}_{kind}.csv'),
                   np.column_stack((lam, sig)), delimiter=',',
                   header='lambda,sigma_over_rho_kBT', comments='')

    # ---------- 参数 JSON 落盘 ----------
    metadata = dict(N=chain1_common.N, xi_f=xi, mu=chain1_common.mu,
                    Delta_E=chain1_common.Delta_E,
                    tail_probability=chain1_common.TAIL,
                    contour_margin=chain1_common.MARGIN,
                    n_state=chain1_common.N_STATE,
                    endpoint_quadrature='geometrically spaced panels, '
                                        '24-point Gauss per panel; refinement 48',
                    G0_finite_reference=res['G0'],
                    gibbs=dict(K1=chain1_common.K1, K2=chain1_common.K2,
                               source='1.png; protein Boltzmann preparation, Gibbs loading',
                               definition='r=x*Lc(f), f=f_MS(x), '
                                          'Lc=N*xi_f*((mu+1)/2+(mu-1)/2*tanh(K1(f-K2)))'),
                    modulus_definition='initial nominal uniaxial slope; '
                                       'shear modulus=G0/3')
    with open(os.path.join(outdir, f'1_chain_N{N}_parameters.json'),
              'w', encoding='utf-8') as f:
        f.write(json.dumps(metadata, indent=2) + '\n')

    # ---------- 绘图（点线图） ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0, 5.0)

    ax.plot(lam_p, sig_p, '--o', color=COLORS['protein'],
            linewidth=lines_linewidth, markersize=MARKERSIZE,
            label=rf'$N={N}$ protein network')
    ax.plot(lam_f, sig_f, '--o', color=COLORS['folded'],
            linewidth=lines_linewidth, markersize=MARKERSIZE,
            label=r'WLC folded')
    ax.plot(lam_u, sig_u, '--o', color=COLORS['unfolded'],
            linewidth=lines_linewidth, markersize=MARKERSIZE,
            label=r'WLC unfolded')
    ax.plot(lam_g, sig_g, '--o', color=COLORS['gibbs'],
            linewidth=lines_linewidth, markersize=MARKERSIZE,
            label=r'Gibbs network model')

    ax.set_xlabel(r'Stretch $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'Stress $\sigma / \rho k_B T$', fontsize=label_fontsize)
    ax.set_title('Constitutive curve', fontsize=title_fontsize, pad=20)

    style_axis(ax)

    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    plt.tight_layout()
    fpath = os.path.join(outdir, f'1_chain_constitutive_N{N}.png')
    fig.savefig(fpath, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('saved:', fpath)


# ===================== 主程序 =====================
def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(outdir, exist_ok=True)
    plot_constitutive_curve(outdir, N=1)


if __name__ == '__main__':
    main()
