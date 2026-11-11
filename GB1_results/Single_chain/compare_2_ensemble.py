#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同质 domain：Gibbs 定力与 Helmholtz 定端距的自由能最小值比较。

物理定义沿用 Gibbs.py / Helmoholtz.py，不改为配分函数热平均。
默认采用 Gibbs.py 的 N、xi_f、alpha、E_mean；同质 domain 均取 E_mean。
Gibbs.py 没有绘图段，本脚本采用与 1_chain_constitutive_curves.py 一致的
线图风格（Times/STIX、内向刻度、双轴刻度、8×6 英寸、300 dpi）。
输出 PNG、CSV 和 JSON 到本脚本旁的 comparison_results/。
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/protein-ensemble-comparison-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent

# ============ 字体与绘图风格设置（与 1_chain_constitutive_curves.py 对齐） ============
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
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 兼容原脚本常量名，最小化改动
FIGSIZE = (8, 6)
SAVE_DPI = savefig_dpi
LINEWIDTH = lines_linewidth
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


# ===================== 物理计算（保持原样，未改动） =====================
def gibbs_defaults():
    """仅读取参数字面值，避免导入 Gibbs.py 时创建其硬编码的外部目录。"""
    wanted = {"N", "xi_f", "alpha", "E_mean", "f_grid_initial", "f_max"}
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # 原脚本绘图字符串中的旧式转义
        tree = ast.parse((HERE/"Gibbs.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    if values.keys() != wanted:
        raise ValueError("Gibbs.py 中缺少所需的常量参数")
    return values


def load_helmholtz(args):
    """复用原有多初值坐标下降；参数只改当前导入实例，不改原脚本。"""
    spec = importlib.util.spec_from_file_location(
        "helmholtz_for_comparison", HERE/"Helmoholtz.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N = args.N
    module.XI_F = args.xi_f
    module.ALPHA = args.alpha
    module.DELTA_E = args.delta_e
    module.A = args.xi_f*(args.alpha-1)
    return module


def gibbs_equilibrium(args, helm):
    """与 min_{r,n}[F_WLC+U-fr] 等价的解析降维全局最小化。

    固定 f 时 x 由 f=f_MS(x) 唯一确定。令 q=x*f-phi(x)，
    剩余 n 势为常数+(DeltaE-A*q)*n-DeltaE*cos(2*pi*n)。
    比较可知全局最小值在 n=0 或 1；A*q=DeltaE 时两者共存。
    """
    A = args.xi_f*(args.alpha-1)

    def x_of_force(f):
        return brentq(lambda x: float(helm.f_ms(x))-f, 0., 1.-1e-12)

    def conjugate(x):
        return x*x/(4*(1-x)**2)+x*x/2

    xc = brentq(lambda x: A*conjugate(x)-args.delta_e, 0., 1.-1e-12)
    fc = float(helm.f_ms(xc))
    force_grid = np.linspace(0., args.force_max, args.force_points)
    rows = []
    for f in force_grid:
        if abs(f-fc) < 1e-10:
            continue
        x = x_of_force(float(f))
        n = 0. if f < fc else 1.
        k = int(args.N*n)
        L = args.N*args.xi_f*(1+(args.alpha-1)*n)
        r = x*L
        F = float(L*helm.phi(x)+args.delta_e*(k-args.N))
        rows.append(dict(r=r, force=f, mean_n=n, n_unfolded=k,
                         L_total=L, F=F, G=F-f*r, coexistence=False))
    if fc <= args.force_max:
        # 有限 N 在同一共存力下有 N+1 种解折叠数，明确保留这些点；
        # 不在跃迁两侧的采样点之间插出一条虚假的倾斜过渡。
        for k in range(args.N+1):
            L = args.N*args.xi_f+A*k
            r = xc*L
            F = float(L*helm.phi(xc)+args.delta_e*(k-args.N))
            rows.append(dict(r=r, force=fc, mean_n=k/args.N, n_unfolded=k,
                             L_total=L, F=F, G=F-fc*r, coexistence=True))
    data = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    return data, dict(force=fc, x=xc, r_folded=xc*args.N*args.xi_f,
                     r_unfolded=xc*args.N*args.alpha*args.xi_f)


def helmholtz_equilibrium(args, helm, gibbs, transition):
    """固定总端距，使用 Helmoholtz.py 原有多初值最低自由能搜索。"""
    all_seeds, starts = helm.make_binary_starts()
    rmax = float(gibbs.r.max())
    rgrid = np.linspace(0., rmax, args.r_points)
    coexist = gibbs.loc[gibbs.coexistence, "r"].to_numpy()
    rgrid = np.unique(np.r_[rgrid, coexist])
    print(f"Helmholtz: {len(all_seeds)} binary seeds -> {len(starts)} symmetry classes",
          flush=True)
    rows = []
    for j, r in enumerate(rgrid):
        n, ri, L, F, force = helm.solve_at_r(float(r), starts)
        k, Fbinary, _ = helm.binary_baseline(float(r))
        if F > Fbinary+1e-7:
            raise AssertionError(("Helmholtz energy above binary candidate", r, F, Fbinary))
        row = dict(r=r, force=force, mean_n=float(n.mean()),
                   variance_n=float(n.var()), L_total=L, F=F,
                   F_binary=Fbinary, k_binary=k)
        for i in range(args.N):
            row[f"n{i+1}"] = n[i]
            row[f"r{i+1}"] = ri[i]
        rows.append(row)
        if j == 0 or (j+1) % 25 == 0 or j+1 == len(rgrid):
            print(f"Helmholtz {j+1}/{len(rgrid)}: r={r:.5g}, f={force:.5g}",
                  flush=True)
    data = pd.DataFrame(rows)
    return data


def validate(args, helm, gibbs, fixed_r, transition):
    """核对解析定力解、恒张力、共存点及 Legendre 自由能下界。"""
    residual = []
    for row in gibbs.itertuples():
        x = row.r/row.L_total
        residual.append(abs(float(helm.f_ms(x))-row.force))
    max_force_residual = max(residual)
    if max_force_residual > 1e-7:
        raise AssertionError(("Gibbs force inversion", max_force_residual))

    # 任意 r 均有 F_H(r)-f_c*r >= min_{r,n}(F-f_c*r)。
    fc = transition["force"]
    xc = transition["x"]
    Gc = args.N*(args.xi_f*float(helm.phi(xc))-args.delta_e-fc*args.xi_f*xc)
    lower_gap = fixed_r.F.to_numpy()-fc*fixed_r.r.to_numpy()-Gc
    if lower_gap.min() < -1e-6:
        raise AssertionError(("Legendre lower bound", lower_gap.min()))
    coexist = gibbs[gibbs.coexistence]
    force_errors = []
    energy_errors = []
    for row in coexist.itertuples():
        match = fixed_r.iloc[int(np.argmin(abs(fixed_r.r-row.r)))]
        force_errors.append(abs(float(match.force)-fc))
        energy_errors.append(abs(float(match.F)-row.F))
    if force_errors and (max(force_errors) > 1e-5 or max(energy_errors) > 1e-6):
        raise AssertionError(("Coexistence check", force_errors, energy_errors))
    if not np.isfinite(fixed_r.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("Nonfinite Helmholtz result")
    n = fixed_r[[f"n{i+1}" for i in range(args.N)]].to_numpy()
    ri = fixed_r[[f"r{i+1}" for i in range(args.N)]].to_numpy()
    Lci = args.xi_f*(1+(args.alpha-1)*n)
    extension_residual = float(np.max(abs(ri.sum(axis=1)-fixed_r.r.to_numpy())))
    tension_residual = float(np.max(abs(helm.f_ms(ri/Lci)-fixed_r.force.to_numpy()[:, None])))
    if np.any(n < -1e-10) or np.any(n > 1+1e-10) or extension_residual > 1e-8 or tension_residual > 1e-7:
        raise AssertionError(("Series constraints", extension_residual, tension_residual))
    return dict(gibbs_max_force_residual=max_force_residual,
                legendre_lower_bound_min_gap=float(lower_gap.min()),
                max_coexistence_force_error=max(force_errors, default=0.),
                max_coexistence_energy_error=max(energy_errors, default=0.),
                helmholtz_binary_bound_max_gap=float((fixed_r.F-fixed_r.F_binary).max()),
                total_extension_residual=extension_residual,
                common_tension_residual=tension_residual,
                helmholtz_solver="original multi-start coordinate descent; not a proof of global optimality")


# ===================== 可视化（风格对齐 1_chain_constitutive_curves.py） =====================
def plot_comparison(args, gibbs, fixed_r, transition):
    """同图展示两个系综；局部放大展开区以看清恒力共存与定端距锯齿。"""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(gibbs.r, gibbs.force, "--o", color='red', lw=LINEWIDTH, ms=MARKERSIZE,
            markevery=4, label="Gibbs ensemble")
    ax.plot(fixed_r.r, fixed_r.force, "--o", color='blue', lw=LINEWIDTH, ms=MARKERSIZE,
            markevery=5, label="Helmholtz ensemble")
    ax.set_xlabel(r"Total extension $r$", fontsize=label_fontsize)
    ax.set_ylabel(r"Force $f$", fontsize=label_fontsize)
    ax.set_title(rf"Ensemble comparison", fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, float(gibbs.r.max()))
    ax.set_ylim(0, 5.0)
    style_axis(ax)
    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')


    fig.tight_layout()
    fig.savefig(args.output/"force_extension_comparison.png", dpi=SAVE_DPI,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


# ===================== 主程序 =====================
def main():
    defaults = gibbs_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults["N"])
    parser.add_argument("--xi-f", type=float, default=defaults["xi_f"])
    parser.add_argument("--alpha", type=float, default=defaults["alpha"])
    parser.add_argument("--delta-e", type=float, default=defaults["E_mean"])
    parser.add_argument("--force-max", type=float, default=defaults["f_max"])
    parser.add_argument("--force-points", type=int, default=defaults["f_grid_initial"])
    parser.add_argument("--r-points", type=int, default=500)
    parser.add_argument("--output", type=Path, default=HERE/"comparison_results")
    args = parser.parse_args()
    if args.N < 1 or args.xi_f <= 0 or args.alpha <= 1 or args.delta_e <= 0 or args.force_max <= 0:
        parser.error("Require N>=1, xi_f>0, alpha>1, delta_e>0, force_max>0")
    if min(args.force_points, args.r_points) < 2:
        parser.error("Require at least two sampling points")
    args.output.mkdir(parents=True, exist_ok=True)
    helm = load_helmholtz(args)
    print(f"Homogeneous: N={args.N}, xi_f={args.xi_f}, alpha={args.alpha}, DeltaE={args.delta_e}",
          flush=True)
    gibbs, transition = gibbs_equilibrium(args, helm)
    print(f"Gibbs coexistence force: {transition['force']:.10g}", flush=True)
    fixed_r = helmholtz_equilibrium(args, helm, gibbs, transition)
    checks = validate(args, helm, gibbs, fixed_r, transition)
    gibbs.to_csv(args.output/"Gibbs_force_extension.csv", index=False)
    fixed_r.to_csv(args.output/"Helmholtz_force_extension.csv", index=False)
    plot_comparison(args, gibbs, fixed_r, transition)
    metadata = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata.update(energy_distribution="homogeneous: all domains use delta_e",
                    equilibrium="free-energy minima, not thermal partition averages",
                    coexistence="N+1 discrete degenerate Gibbs minima; connecting line is a visual guide",
                    transition=transition, validation=checks,
                    figure_style="8x6 inches, 300 dpi, Times/STIX; aligned with 1_chain_constitutive_curves.py")
    (args.output/"parameters.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Saved: "+str(args.output/"force_extension_comparison.png"), flush=True)


if __name__ == "__main__":
    main()