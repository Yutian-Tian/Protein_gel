"""
================================================================================
 3-chain 网络本构曲线可视化（3_chain_constitutive_curves.py）

 模型计算全部来自 3_chain_common.py，本脚本只负责：
   · 调用 compute_constitutive() 得到四模型应力曲线
   · 顺序检查（protein ≥ WLC unfolded）与 Gibbs 数值验证
   · 点线图绘制、CSV/验证文本落盘（只输出 PNG）

 直接运行：python 3_chain_constitutive_curves.py
 输出位置：同目录 Figures/Constitutive_Curve_N*.png
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import cumulative_trapezoid

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
    """计算并绘制 N 段蛋白网络、两种 WLC 基线与 Gibbs 网络的本构曲线。"""
    # ---------- 模型计算（3_chain_common.py） ----------
    print(f'=== N = {N} 四模型本构 ===')
    res = chain3_common.compute_constitutive(N=N)
    lam, sig = res['lam'], res['sig']
    lam_f, sig_f = res['lam_f'], res['sig_f']
    lam_u, sig_u = res['lam_u'], res['sig_u']
    sig_g = res['sig_g']
    ch = res['chain']

    print(f'支撑区间 [0, {ch.r[-1]:.3f}] lp，有效网格点数 {len(ch.r)}')
    print(f'R0 = {ch.R0:.4f} lp, E0 = {ch.E0():.4f} ρkBT')
    print(f'lambda_max = {res["lam_max"]:.2f}, 末端应力 = {sig[-1]:.3f}')
    print(f'[WLC folded]   L = {res["wlc_folded"].L:.2f} lp, '
          f'E0 = {res["wlc_folded"].E0():.4f} ρkBT')
    print(f'[WLC unfolded] L = {res["wlc_unfolded"].L:.2f} lp, '
          f'E0 = {res["wlc_unfolded"].E0():.4f} ρkBT')
    print(f'[Gibbs] k1={chain3_common.K1}, k2={chain3_common.K2}, '
          f'E0={res["gibbs"].E0():.6f} rho*kBT')

    # ---------- 顺序检查与 Gibbs 验证 ----------
    gap = sig - sig_u
    print(f'min(sigma_protein - sigma_unfolded) = {gap.min():.8g}')
    if gap.min() < -1e-8:
        raise RuntimeError('Protein/WLC stress ordering failed; '
                           'inspect model and quadrature')

    validation = chain3_common.validate_gibbs(res['gibbs'])
    for line in validation:
        print(line)
    with open(os.path.join(outdir, f'Gibbs_N{N}_validation.txt'), 'w') as report:
        report.write(f'N={N}, xi_f={chain3_common.xi_f}, mu={chain3_common.mu}, '
                     f'Delta_E={chain3_common.Delta_E}, '
                     f'k1={chain3_common.K1}, k2={chain3_common.K2}, '
                     f'tail={chain3_common.WTOL}, DR={chain3_common.DR}\n')
        report.write('\n'.join(validation) + '\n')

    # ---------- CSV 落盘 ----------
    np.savetxt(os.path.join(outdir, f'Constitutive_Curve_N{N}_gibbs.csv'),
               np.column_stack((lam, sig, sig_u, sig_g)), delimiter=',',
               header='lambda,sigma_protein,sigma_WLC_unfolded,sigma_gibbs',
               comments='')
    # 零点为 lambda=1；通过沿应力曲线积分保存相对网络能量。
    energies = [cumulative_trapezoid(stress, lam, initial=0)
                for stress in (sig, sig_u, sig_g)]
    np.savetxt(os.path.join(outdir, f'Network_energy_N{N}_gibbs.csv'),
               np.column_stack((lam, *energies)), delimiter=',',
               header='lambda,deltaF_protein,deltaF_WLC_unfolded,deltaF_gibbs',
               comments='')
    np.savetxt(os.path.join(outdir, f'Constitutive_Curve_N{N}_gibbs_baselines.csv'),
               np.column_stack((lam, sig, sig_u)), delimiter=',',
               header='lambda,sigma_protein_over_rho_kBT,sigma_unfolded_over_rho_kBT',
               comments='')
    np.savetxt(os.path.join(outdir, f'Constitutive_Curve_N{N}_gibbs_folded.csv'),
               np.column_stack((lam_f, sig_f)), delimiter=',',
               header='lambda,sigma_folded_over_rho_kBT', comments='')

    # ---------- 绘图（点线图） ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0, 5.0)

    ax.plot(lam, sig, '--o', color='red', linewidth=lines_linewidth,
            markersize=MARKERSIZE,
            label=rf'$N={N}$ protein network')
    ax.plot(lam_f, sig_f, '--s', color='blue', linewidth=lines_linewidth,
            markersize=MARKERSIZE,
            label=r'WLC folded')
    ax.plot(lam_u, sig_u, '--^', color='green', linewidth=lines_linewidth,
            markersize=MARKERSIZE,
            label=r'WLC unfolded')
    ax.plot(lam, sig_g, '-.D', color='black', linewidth=lines_linewidth,
            markersize=MARKERSIZE-1,
            label=r'Gibbs network model')

    ax.set_xlabel(r'Stretch $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'Stress $\sigma / \rho k_B T$', fontsize=label_fontsize)
    ax.set_title('Constitutive curve', fontsize=title_fontsize, pad=20)

    style_axis(ax)

    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    plt.tight_layout()
    fpath = os.path.join(outdir, f'Constitutive_Curve_N{N}.png')
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
