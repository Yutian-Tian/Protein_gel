"""N=1 各向同性仿射（一链）网络：单文件独立计算程序。

依赖：Python >= 3.10，NumPy >= 1.24，SciPy >= 1.10，Matplotlib >= 3.6。
安装：python -m pip install numpy scipy matplotlib
直接运行：python 1_chain_networks.py
全部任务（含尾部截断比较）：python 1_chain_networks.py --task all
只画条件分布：python 1_chain_networks.py --task conditional
只做尾部截断比较：python 1_chain_networks.py --tail-sensitivity
输出目录：--output-dir PATH；默认位于本文件旁 Figures/1_chain_standalone。

本文件包含所有物理模型、积分、验证及绘图函数，不依赖其他项目脚本或已有CSV。
默认生成三种网络本构、初始模量扫描、完整参考分布及p(n|r_peak)。
约化单位kBT=lp=1；rho为总链密度，sigma为名义应力。
绘图风格与四模型本构代码逐项对齐：局部 rc_context + Times New Roman、
figsize=(8,6)、linewidth=3、legend(fontsize=19)、grid(alpha=0.3)、
线型 b--/r-/g--、label "$N=1$ protein network"、ax.set(...) 一次设置、
savefig(dpi=300) 无 bbox_inches。
"""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import os
import platform
import time

import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq
from scipy.special import logsumexp

# ==================== 参数（与原有一链计算一致） ====================
N = 1
XI_F = 3.6
mu = 7.6
Delta_E = 11.9
TAIL = 0.005
TAIL_FRACTIONS = (0.005, 0.01, 0.02, 0.05)
BASE_MARGIN = 0.995
MARGIN = 1.0 - 1e-8
N_STATE = 512
N_RADIAL = 256
N_ANGLE = 48
OUTDIR = Path(__file__).resolve().parent / 'Figures' / '1_chain_standalone'

# 线型与 label 与四模型本构代码一致
STYLES = {'folded': 'b--', 'protein': 'r-', 'unfolded': 'g--'}
LABELS = {'folded': 'WLC folded',
          'protein': f'$N={N}$ protein network',
          'unfolded': 'WLC unfolded'}
TRAPEZOID = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ==================== 绘图样式（向代码2完全看齐） ====================
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
ytick_right = True

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

PLOT_STYLE = {
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
}

LINE_WIDTH = lines_linewidth
LEGEND_FONTSIZE = legend_fontsize
LEGEND_TITLE_FONTSIZE = legend_title_fontsize
GRID_ALPHA = grid_alpha
SAVEFIG_DPI = savefig_dpi


# ==================== 单链自由能与退火统计 ====================
def phi(x):
    """WLC 自由能的无量纲部分"""
    return x**2 * (3 - 2*x) / (4 * (1 - x))


def f_MS(x):
    """微观力解析计算 (beta=1, lp=1)"""
    return 0.25 * (1 - x)**(-2) - 0.25 + x


def fcp(x):
    """微观力对 x 的解析导数"""
    return 0.5 * (1 - x)**(-3) + 1.0


def Lc_of_n(n, N, xi_f):
    """当前内部状态n对应的轮廓长度。"""
    return N * xi_f * (1 + n*(mu - 1))


def U_of_n(n):
    """内部变量的势能，单位kBT。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)


@lru_cache(maxsize=None)
def gauss_unit(points):
    q, w = np.polynomial.legendre.leggauss(points)
    return (q + 1) / 2, w / 2


def annealed_statistics(r, N, xi_f, n_points=512):
    """logZ、平均力、退火刚度、固定态平均刚度及条件力方差。"""
    r = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("End distance must be finite and nonnegative")
    lower = np.maximum(0., (r / (N * xi_f) - 1) / (mu - 1))
    if np.any(lower >= 1):
        raise ValueError('End distance must remain below maximum contour length')
    t, wn = gauss_unit(n_points)
    n = lower[..., None] + (1 - lower[..., None]) * t
    Lc = Lc_of_n(n, N, xi_f)
    x = r[..., None] / Lc
    lw = -Lc * phi(x) - U_of_n(n) + np.log((1 - lower[..., None]) * wn)
    logZ = logsumexp(lw, axis=-1)
    conditional = np.exp(lw - logZ[..., None])
    fn = f_MS(x)
    f = np.sum(conditional * fn, axis=-1)
    frozen_fp = np.sum(conditional * fcp(x) / Lc, axis=-1)
    variance = np.sum(conditional * (fn - f[..., None])**2, axis=-1)
    return logZ, f, frozen_fp - variance, frozen_fp, variance


def build_annealed(N, xi_f, n_points=512, r_points=1024):
    # r=Lmax*t²：保留完整支撑，并在小端距区域增加节点。
    t, wt = gauss_unit(r_points)
    Lmax = N * xi_f * mu
    r = Lmax * t**2
    wr = wt * (2 * Lmax * t)
    logZ, f, fp, frozen_fp, variance = annealed_statistics(r, N, xi_f, n_points)
    logmass = logZ + 2 * np.log(r) + np.log(wr)
    mass = np.exp(logmass - logsumexp(logmass))
    return dict(r=r, mass=mass, f=f, fp=fp, frozen_fp=frozen_fp,
                variance=variance, N=N, xi_f=xi_f, n_points=n_points)


# ==================== 一链网络及数值验证 ====================
def response(r, xi, kind, order=N_STATE):
    if kind == 'protein':
        logz, f, fp, _, _ = annealed_statistics(r, 1, xi, order)
        return -logz, f, fp
    L = xi if kind == 'folded' else mu*xi
    x = np.asarray(r)/L
    if np.any(x >= 1) or np.any(x < 0):
        raise ValueError('Chain distance outside contour length')
    return L*phi(x), f_MS(x), fcp(x)/L


def finite_reference(xi, kind, points=N_RADIAL, tail=TAIL):
    """先固定截断制备分布，再生成径向求积节点，不随lambda删链。"""
    L = xi if kind == 'folded' else mu*xi
    rg = np.linspace(0, L, max(2001, int(L/.005)+1))
    F = response(rg[:-1], xi, 'folded' if kind == 'folded' else 'protein')[0]
    logz = -F
    p = np.append(rg[:-1]**2*np.exp(logz-logz.max()), 0.)
    cdf = cumulative_trapezoid(p, rg, initial=0.)
    cdf /= cdf[-1]
    cutoff = np.interp(1-tail, cdf, rg)
    t, wt = gauss_unit(points)
    r, wr = cutoff*t**2, 2*cutoff*t*wt
    logz = -response(r, xi, 'folded' if kind == 'folded' else 'protein')[0]
    lw = 2*np.log(r)+logz+np.log(wr)
    mass = np.exp(lw-logsumexp(lw))
    return dict(r=r, mass=mass, cutoff=cutoff, L=L)


def endpoint_nodes(gap, order=24):
    """按到端点的距离分几何面板，解析接近轮廓时的尖锐贡献。"""
    edges=np.unique(np.r_[0., np.geomspace(gap*.1,1.,
        max(2,int(np.ceil(-np.log10(gap*.1)))+1))])
    t,w=gauss_unit(order)
    distance=np.concatenate([a+(b-a)*t for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(b-a)*w for a,b in zip(edges[:-1],edges[1:])])
    return 1-distance,weights


def constitutive_grid(pre):
    """原始区域保持120点；轮廓附近按剩余间隙对数加密。"""
    limit=pre['L']/pre['cutoff']
    base=np.linspace(1.,BASE_MARGIN*limit,120)
    extended=(1-np.geomspace(1-BASE_MARGIN,1-MARGIN,21)[1:])*limit
    return np.r_[base,extended]


def orient_average(pre, xi, kind, lam, angles=N_ANGLE, energy=False, state_order=N_STATE):
    gap=1-lam*pre['cutoff']/pre['L']
    if gap <= 0:
        raise ValueError('Retained reference support reaches the contour length')
    if gap < 1-BASE_MARGIN-1e-12:
        # 相同的固定参考密度重新求积；没有随拉伸删链。
        v,wv=endpoint_nodes(gap, max(24, angles//2))
        r=pre['cutoff']*v
        reference_kind='folded' if kind=='folded' else 'protein'
        logz=-response(r,xi,reference_kind)[0]
        logmass=2*np.log(r)+logz+np.log(pre['cutoff']*wv)
        pre=dict(pre,r=r,mass=np.exp(logmass-logsumexp(logmass)))
        u,wu=endpoint_nodes(gap,max(24,angles//2))
    else:
        u, wu = gauss_unit(angles)
    stretch = np.sqrt(lam**2*u*u+lam**-1*(1-u*u))
    derivative = (lam*u*u-.5*lam**-2*(1-u*u))/stretch
    total = 0.
    # 各块直接积分内部态；避免r×u×n完整张量常驻内存。
    for start in range(0, len(pre['r']), 24):
        r = pre['r'][start:start+24]
        mass = pre['mass'][start:start+24]
        F, f, _ = response(r[:, None]*stretch, xi, kind, order=state_order)
        values = F if energy else r[:, None]*f*derivative
        total += np.sum(mass*np.sum(values*wu,axis=-1))
    return total


def initial_modulus(pre, xi, kind):
    r,w = pre['r'],pre['mass']
    _,f,fp = response(r,xi,kind)
    return np.sum(w*(r*r*fp+4*r*f))/5


def full_reference(xi, kind):
    if kind != 'folded':
        return build_annealed(1,xi)
    x,wx = gauss_unit(1024)
    lw = 2*np.log(x)-xi*phi(x)+np.log(wx)
    return dict(r=xi*x,mass=np.exp(lw-logsumexp(lw)))


def validate(pre_protein, pre_folded):
    lines=[]
    u,w = gauss_unit(N_ANGLE)
    a=(3*u*u-1)/2
    assert abs(np.sum(w*a))<1e-12
    assert np.isclose(np.sum(w*a*a),.2,atol=1e-12)
    # 高斯网络：<r²>=3，f=r，解析sigma=lambda-lambda^-2，G0=3。
    for lm in [1.,1.5,3.]:
        stretch=np.sqrt(lm*lm*u*u+lm**-1*(1-u*u))
        d=(lm*u*u-.5*lm**-2*(1-u*u))/stretch
        assert np.isclose(3*np.sum(w*stretch*d),lm-lm**-2,atol=1e-12)
    for kind in LABELS:
        pre=pre_folded if kind=='folded' else pre_protein
        g=initial_modulus(pre,XI_F,kind)
        h=1e-5
        slope=(orient_average(pre,XI_F,kind,1+h)-
               orient_average(pre,XI_F,kind,1-h))/(2*h)
        assert abs(orient_average(pre,XI_F,kind,1.))<1e-10
        assert np.isclose(g,slope,rtol=2e-6,atol=1e-7),(kind,g,slope)
        lm=1+.6*(MARGIN*pre['L']/pre['cutoff']-1)
        stress=orient_average(pre,XI_F,kind,lm)
        dF=(orient_average(pre,XI_F,kind,lm+h,energy=True)-
            orient_average(pre,XI_F,kind,lm-h,energy=True))/(2*h)
        assert np.isclose(stress,dF,rtol=2e-6,atol=1e-7)
        high=orient_average(pre,XI_F,kind,lm,angles=96)
        assert np.isclose(stress,high,rtol=2e-4,atol=1e-7),(kind,stress,high)
        refined=finite_reference(XI_F,kind,points=512)
        sr=orient_average(refined,XI_F,kind,lm,angles=96)
        assert np.isclose(stress,sr,rtol=2e-4,atol=1e-7),(kind,stress,sr)
        endpoint=MARGIN*pre['L']/pre['cutoff']
        end_sigma=orient_average(pre,XI_F,kind,endpoint)
        end_refined=orient_average(pre,XI_F,kind,endpoint,angles=96)
        assert np.isclose(end_sigma,end_refined,rtol=2e-4,atol=1e-7)
        lines.append(f'{kind}: endpoint_sigma={end_sigma:.10g}, endpoint_refined={end_refined:.10g}')
        line=(f'{kind}: G0_finite_reference={g:.10g}, sigma_derivative={slope:.10g}, '
              f'lambda={lm:.6g}, sigma={stress:.10g}, dF={dF:.10g}, refined={sr:.10g}')
        print(line,flush=True);lines.append(line)
    for xi in [1.,3.6,100.]:
        pre=build_annealed(1,xi)
        fine=build_annealed(1,xi,n_points=1024,r_points=2048)
        for kind in ['protein','unfolded']:
            g=initial_modulus(pre,xi,kind);gf=initial_modulus(fine,xi,kind)
            assert np.isclose(g,gf,rtol=2e-6,atol=1e-7),(xi,kind,g,gf)
            lines.append(f'full reference xi_f={xi:g}, {kind}: G0={g:.10g}, refined={gf:.10g}')
    return lines


# ==================== 尾部截断比较 ====================
def tail_sensitivity():
    """比较固定截去0.5%、1%、2%、5%概率的有限参考系综；不改模量图。"""
    OUTDIR.mkdir(exist_ok=True)
    records=[]; checks=[]
    colors = plt.get_cmap('viridis')(np.linspace(.05, .9, len(TAIL_FRACTIONS)))

    # 先算数据（与绘图无关）
    sig_by_tail = {}
    for j, tail in enumerate(TAIL_FRACTIONS):
        percent=100*tail
        tag=f'{percent:g}pct'
        protein=finite_reference(XI_F,'protein',tail=tail)
        folded=finite_reference(XI_F,'folded',tail=tail)
        sig_by_tail[tail] = {}
        for kind in LABELS:
            pre=folded if kind=='folded' else protein
            target=constitutive_grid(pre)
            path=OUTDIR/f'1_chain_tail_{tag}_{kind}.csv'
            sig=[]
            for i,lm in enumerate(target):
                order=N_STATE if tail==TAIL else 256
                sig.append(orient_average(pre,XI_F,kind,lm,state_order=order))
                if i%35==0:
                    print(f'tail={percent:g}%, {kind}: {i+1}/{len(target)}',flush=True)
            sig=np.asarray(sig)
            assert np.isfinite(sig).all() and abs(sig[0])<1e-10
            # 实际高伸长节点对比加密方向积分和512内部态节点。
            for idx in [len(target)//2,len(target)-1]:
                fine=orient_average(pre,XI_F,kind,target[idx],angles=96,state_order=512)
                relative=abs(sig[idx]-fine)/max(abs(fine),1e-12)
                assert relative<1e-3,(tail,kind,idx,sig[idx],fine,relative)
                checks.append(f'tail={tail:g}, {kind}, lambda={target[idx]:.10g}, '
                              f'sigma={sig[idx]:.10g}, refined={fine:.10g}, rel={relative:.6g}')
            np.savetxt(path,np.column_stack((target,sig)),delimiter=',',
                       header='lambda,sigma_over_rho_kBT',comments='')
            records.append([tail, list(LABELS).index(kind),pre['cutoff'],
                            pre['L']/pre['cutoff'],target[-1],sig[-1],
                            initial_modulus(pre,XI_F,kind)])
            sig_by_tail[tail][kind] = (target, sig)
            print(f'tail={percent:g}%, {kind}: lambda_max={target[-1]:.6g}, '
                  f'sigma_end={sig[-1]:.6g}',flush=True)

    # 绘图：2×2 总览 + 1×2 对比（多子图保留较大 figsize）
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        compare, axs = plt.subplots(1, 2, figsize=(14, 6))
        for j, tail in enumerate(TAIL_FRACTIONS):
            percent = 100*tail
            ax = axes.flat[j]
            for kind in LABELS:
                target, sig = sig_by_tail[tail][kind]
                ax.plot(target, sig, STYLES[kind], label=LABELS[kind],
                        linewidth=LINE_WIDTH)
                if kind in ['protein', 'unfolded']:
                    k = 0 if kind == 'protein' else 1
                    axs[k].plot(target, sig, color=colors[j],
                                label=f'{percent:g}%', linewidth=LINE_WIDTH)
            ax.set(title=f'Removed probability: {percent:g}%',
                   xlabel=r'Stretch $\lambda$',
                   ylabel=r'$\sigma/(\rho k_B T)$', ylim=(0, 5))
            ax.grid(alpha=GRID_ALPHA)
            ax.legend(fontsize=14)
        fig.tight_layout()
        fig.savefig(OUTDIR/'1_chain_tail_sensitivity.png', dpi=SAVEFIG_DPI)
        plt.close(fig)
        for k, kind in enumerate(['protein', 'unfolded']):
            axs[k].set(title=LABELS[kind], xlabel=r'Stretch $\lambda$',
                       ylabel=r'$\sigma/(\rho k_B T)$', ylim=(0, 5))
            axs[k].grid(alpha=GRID_ALPHA)
            axs[k].legend(title='Removed probability', fontsize=14, title_fontsize=14)
        compare.tight_layout()
        compare.savefig(OUTDIR/'1_chain_tail_sensitivity_comparison.png',
                        dpi=SAVEFIG_DPI)
        plt.close(compare)

    np.savetxt(OUTDIR/'1_chain_tail_summary.csv',records,delimiter=',',
               header='tail_probability,model_id_0folded_1protein_2unfolded,r_cut_over_lp,lambda_geometric_limit,lambda_calculated_end,sigma_end_over_rho_kBT,G0_finite_reference_over_rho_kBT',comments='')
    (OUTDIR/'1_chain_tail_validation.txt').write_text('\n'.join(checks)+'\n')
    print('Tail sensitivity comparison saved.',flush=True)


# ==================== 初始模量扫描 ====================
def plot_initial_modulus(xis, values):
    """绘制初始模量，风格与四模型本构代码一致。"""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        for j, kind in enumerate(['folded', 'protein']):
            ax.plot(xis, values[:, j], STYLES[kind], label=LABELS[kind],
                    linewidth=LINE_WIDTH)
        ax.axhline(3, color='black', linestyle='--', linewidth=2.5,
                   label='Rubber modulus = 3')
        ax.set(xlabel=r'Folded length $\xi_f/l_p$',
               ylabel=r'$G_0/(\rho k_B T)$',
               title='Initial modulus: isotropic affine network',
               xlim=(1, 100), ylim=(0, 1.08 * values.max()))
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        fig.tight_layout()
        fig.savefig(OUTDIR/'1_chain_G0_vs_xi_f_N1.png', dpi=SAVEFIG_DPI)
        plt.close(fig)


# ==================== 参考态峰及条件分布 ====================
def log_weight(n, r, xi):
    """保留轮廓可达域；不可达状态的 Boltzmann 权重严格为零。"""
    n = np.asarray(n, dtype=float)
    contour = Lc_of_n(n, N, xi)
    valid = (n >= 0) & (n <= 1) & (r < contour)
    values = np.full(n.shape, -np.inf)
    values[valid] = (-contour[valid] * phi(r / contour[valid])
                     - U_of_n(n[valid]))
    return values


def reference_peak(xi, state_order=N_STATE):
    """扫描完整支撑的各个峰，以 r*f_eff(r)=2 精化，再选择全局峰。"""
    contour = N * mu * xi
    grid = np.linspace(0., contour, 4097)
    logz, force, *_ = annealed_statistics(grid[:-1], N, xi, state_order)
    stationary = grid[:-1] * force - 2.
    crossings = np.flatnonzero((stationary[:-1] <= 0) & (stationary[1:] > 0))
    if not len(crossings):
        raise RuntimeError('No reference-distribution peak found')

    def residual(r):
        f = annealed_statistics(r, N, xi, state_order)[1]
        return float(r * f - 2.)

    peaks = [brentq(residual, grid[i], grid[i+1], xtol=1e-13)
             for i in crossings]
    scores = [2*np.log(r) + float(annealed_statistics(
        r, N, xi, state_order)[0]) for r in peaks]
    peak = peaks[int(np.argmax(scores))]
    return peak, peaks, grid, logz


def reference_log_normalization(xi, radial_order):
    t, weights = gauss_unit(radial_order)
    contour = N * mu * xi
    r = contour * t**2
    logz = annealed_statistics(r, N, xi, N_STATE)[0]
    return float(logsumexp(2*np.log(r) + logz
                          + np.log(2*contour*t*weights)))


def conditional_distribution(xi):
    if not np.isfinite(xi) or xi <= 0:
        raise ValueError('xi_f must be positive and finite')
    OUTDIR.mkdir(exist_ok=True)
    peak, candidates, rg, logz_grid = reference_peak(xi)
    peak_refined = reference_peak(xi, state_order=1024)[0]
    logz_peak, force_peak, *_ = annealed_statistics(peak, N, xi, N_STATE)
    logz_peak = float(logz_peak)
    logz_refined = float(annealed_statistics(peak, N, xi, 1024)[0])
    lognorm = reference_log_normalization(xi, 1024)
    lognorm_refined = reference_log_normalization(xi, 2048)

    n = np.linspace(0., 1., 10001)
    density = np.exp(log_weight(n, peak, xi) - logz_peak)
    lower = max(0., (peak/(N*xi)-1.)/(mu-1.))

    def conditional(s):
        return float(np.exp(log_weight(s, peak, xi) - logz_peak))

    integral, integral_error = quad(conditional, lower, 1.,
                                    epsabs=1e-12, epsrel=1e-12)
    mean, _ = quad(lambda s: s*conditional(s), lower, 1.,
                   epsabs=1e-12, epsrel=1e-12)
    folded_mass, _ = quad(conditional, lower, max(lower, .5),
                          epsabs=1e-12, epsrel=1e-12)
    csv_integral = float(TRAPEZOID(density, n))
    residual = float(peak*force_peak-2.)
    checks = {
        'conditional_integral_adaptive': integral,
        'conditional_integral_error_estimate': integral_error,
        'conditional_integral_csv_trapezoid': csv_integral,
        'peak_stationarity_r_times_f_minus_2': residual,
        'r_peak_state_order_1024': peak_refined,
        'r_peak_refinement_absolute_change': abs(peak-peak_refined),
        'log_Z_conditional_refinement_absolute_change': abs(logz_peak-logz_refined),
        'log_Z_reference_radial_refinement_absolute_change': abs(lognorm-lognorm_refined),
    }
    assert abs(integral-1.) < 1e-10, checks
    assert abs(csv_integral-1.) < 2e-6, checks
    assert abs(residual) < 1e-10, checks
    assert abs(peak-peak_refined) < 1e-8, checks
    assert abs(logz_peak-logz_refined) < 1e-9, checks
    assert abs(lognorm-lognorm_refined) < 1e-8, checks

    p0 = np.zeros_like(rg)
    p0[1:-1] = np.exp(2*np.log(rg[1:-1])+logz_grid[1:]-lognorm)
    p0_peak = float(np.exp(2*np.log(peak)+logz_peak-lognorm))
    stem = f'1_chain_Conditional_distribution_N{N}_xi{xi:g}'
    np.savetxt(OUTDIR/f'{stem}.csv', np.column_stack((n, density)),
               delimiter=',', header='n,p_n_given_r_peak', comments='')
    np.savetxt(OUTDIR/f'{stem}_reference.csv', np.column_stack((rg, p0)),
               delimiter=',', header='r_over_lp,p0_protein_full', comments='')

    # 与四模型本构代码一致的绘图风格
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        label = rf'$r_{{\mathrm{{peak}}}}/l_p={peak:.4f}$'
        visible = density > 0
        ax.plot(n[visible], density[visible], color='purple',
                linewidth=LINE_WIDTH, linestyle='-',
                label=label, marker='o', markevery=200, markersize=6,
                markerfacecolor='purple', markeredgewidth=1.2)
        ax.set(yscale='log', xlim=(0, 1),
               xlabel=r'Unfolded fraction $n$',
               ylabel=r'Probability density $p(n\mid r_{\mathrm{peak}})$')
        ax.set_title(r'Conditional Distribution ($N=1$)', pad=20)
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        fig.tight_layout()
        for suffix in ['png', 'pdf']:
            fig.savefig(OUTDIR/f'{stem}.{suffix}', dpi=SAVEFIG_DPI)
        plt.close(fig)

    metadata = {
        'N': N, 'xi_f_over_lp': xi, 'mu': mu,
        'Delta_E_over_kBT': Delta_E,
        'reference_distribution': 'p0(r) = r^2 Z(r) / integral r^2 Z(r) dr',
        'conditional_distribution': 'p(n|r_peak) = exp(-F(r_peak,n)) / Z(r_peak)',
        'conditional_measure': 'uniform dn on accessible states in [0,1]',
        'tail_probability_removed': 0.,
        'r_peak_over_lp': peak, 'candidate_peak_positions_over_lp': candidates,
        'p0_at_peak_times_lp': p0_peak,
        'mean_n_given_r_peak': mean,
        'probability_n_below_half_given_r_peak': folded_mass,
        'probability_n_above_half_given_r_peak': 1.-folded_mass,
        'validation': checks,
    }
    (OUTDIR/f'{stem}_parameters.json').write_text(
        json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(f'Saved figure, CSV data and parameters: {OUTDIR / stem}', flush=True)
    return metadata


# ==================== 各计算任务 ====================
def constitutive_curves():
    """固定截断参考态，生成三种网络本构并进行数值验证。"""
    protein = finite_reference(XI_F, 'protein')
    folded = finite_reference(XI_F, 'folded')
    validation = validate(protein, folded)
    (OUTDIR/'1_chain_N1_validation.txt').write_text(
        '\n'.join(validation)+'\n', encoding='utf-8')
    print('Validation passed. Computing constitutive curves.', flush=True)
    slopes = {}
    curves = {}
    for kind in LABELS:
        pre = folded if kind == 'folded' else protein
        lam = constitutive_grid(pre)
        sig = []
        for i, lm in enumerate(lam):
            sig.append(orient_average(pre, XI_F, kind, lm))
            if i % 35 == 0:
                print(f'constitutive {kind}: {i+1}/{len(lam)}', flush=True)
        sig = np.asarray(sig)
        curves[kind] = (lam, sig)
        slopes[kind] = float(initial_modulus(pre, XI_F, kind))
        np.savetxt(OUTDIR/f'1_chain_constitutive_N1_{kind}.csv',
                   np.column_stack((lam, sig)), delimiter=',',
                   header='lambda,sigma_over_rho_kBT', comments='')
        print(f'{kind}: lambda_end={lam[-1]:.10g}, sigma_end={sig[-1]:.10g}',
              flush=True)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        for kind in LABELS:
            lam, sig = curves[kind]
            ax.plot(lam, sig, STYLES[kind], label=LABELS[kind],
                    linewidth=LINE_WIDTH)
        ax.set(xlabel=r'Stretch $\lambda$',
               ylabel=r'$\sigma/(\rho k_B T)$',
               title='Isotropic affine network',
               xlim=(1, 8), ylim=(0, 5))
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        fig.tight_layout()
        fig.savefig(OUTDIR/'1_chain_constitutive_N1.png', dpi=SAVEFIG_DPI)
        plt.close(fig)

    full_initial = {kind: float(initial_modulus(full_reference(XI_F, kind), XI_F, kind))
                    for kind in ['folded', 'protein']}
    metadata = dict(N=N, xi_f=XI_F, mu=mu, Delta_E=Delta_E,
                    tail_probability=TAIL, contour_margin=MARGIN,
                    n_state=N_STATE, n_radial=N_RADIAL, n_angle=N_ANGLE,
                    endpoint_quadrature='geometrically spaced panels, 24-point Gauss per panel; refinement 48',
                    G0_finite_reference=slopes, G0_full_reference=full_initial,
                    modulus_definition='initial nominal uniaxial slope; shear modulus=G0/3')
    (OUTDIR/'1_chain_N1_parameters.json').write_text(
        json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def modulus_scan():
    """使用完整参考分布计算xi_f/lp=1..100的初始名义拉伸模量。"""
    xis = np.linspace(1, 100, 100)
    values = []
    for i, xi in enumerate(xis):
        pp = full_reference(xi, 'protein')
        pf = full_reference(xi, 'folded')
        values.append([initial_modulus(pf, xi, 'folded'),
                       initial_modulus(pp, xi, 'protein')])
        if i % 10 == 0:
            print(f'modulus scan: {i+1}/{len(xis)}', flush=True)
    values = np.asarray(values)
    np.savetxt(OUTDIR/'1_chain_G0_vs_xi_f_N1.csv', np.column_stack((xis, values)),
               delimiter=',', header='xi_f_over_lp,G0_folded,G0_protein', comments='')
    plot_initial_modulus(xis, values)


def reference_distribution():
    """完整参考分布；解折叠WLC继承蛋白参考态，折叠WLC采用自身参考态。"""
    contour = mu*N*XI_F
    r = np.linspace(0., contour, max(2001, int(contour/.005)+1))
    logz = annealed_statistics(r[:-1], N, XI_F, N_STATE)[0]
    lognorm = reference_log_normalization(XI_F, 1024)
    protein = np.zeros_like(r)
    protein[1:-1] = np.exp(2*np.log(r[1:-1])+logz[1:]-lognorm)
    rf = np.linspace(0., N*XI_F, max(2001, int(N*XI_F/.005)+1))
    x, wx = gauss_unit(1024)
    lf = N*XI_F
    norm_folded = logsumexp(2*np.log(lf*x)-lf*phi(x)+np.log(lf*wx))
    folded = np.zeros_like(rf)
    folded[1:-1] = np.exp(2*np.log(rf[1:-1])-lf*phi(rf[1:-1]/lf)-norm_folded)
    np.savetxt(OUTDIR/'1_chain_Reference_distribution.csv',
               np.column_stack((r, protein, protein)), delimiter=',',
               header='r_over_lp,p0_protein_full,p0_WLC_unfolded_full',
               comments='')
    np.savetxt(OUTDIR/'1_chain_Reference_distribution_folded.csv',
               np.column_stack((rf, folded)), delimiter=',',
               header='r_over_lp,p0_WLC_folded_full', comments='')

    # 线型与 label 复用 STYLES / LABELS，与四模型本构代码一致
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r, protein, STYLES['protein'], label=LABELS['protein'],
                linewidth=LINE_WIDTH)
        ax.plot(rf, folded, STYLES['folded'], label=LABELS['folded'],
                linewidth=LINE_WIDTH)
        ax.plot(r, protein, STYLES['unfolded'], label=LABELS['unfolded'],
                linewidth=LINE_WIDTH)
        ax.set(xlim=(0, 10),
               xlabel=r'End-to-end distance $r\,/\,l_p$',
               ylabel=r'$p_0(r)$')
        ax.set_title('Reference state distribution', pad=20)
        ax.grid(alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        fig.tight_layout()
        fig.savefig(OUTDIR/'1_chain_Reference_distribution.png', dpi=SAVEFIG_DPI)
        plt.close(fig)


def validation_only():
    results = validate(finite_reference(XI_F, 'protein'),
                       finite_reference(XI_F, 'folded'))
    (OUTDIR/'1_chain_N1_validation.txt').write_text(
        '\n'.join(results)+'\n', encoding='utf-8')
    print('All network validation checks passed.', flush=True)


# ==================== 主程序 ====================
def run():
    global XI_F, OUTDIR
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task',
                        choices=['core', 'all', 'constitutive', 'modulus',
                                 'reference', 'conditional', 'tail', 'validate'],
                        default='core',
                        help='default: core (constitutive, modulus, reference, conditional)')
    parser.add_argument('--tail-sensitivity', action='store_true',
                        help='run only tail-sensitivity comparison')
    parser.add_argument('--xi-f', type=float, default=XI_F,
                        help='folded contour length in lp; modulus scan remains 1..100')
    parser.add_argument('--output-dir', type=Path, default=OUTDIR,
                        help='output directory, created if needed')
    args = parser.parse_args()
    if not np.isfinite(args.xi_f) or args.xi_f <= 0:
        parser.error('--xi-f must be positive and finite')
    if args.tail_sensitivity and args.task not in ['core', 'tail']:
        parser.error('--tail-sensitivity cannot be combined with this --task')
    XI_F = args.xi_f
    OUTDIR = args.output_dir.expanduser().resolve()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    selection = 'tail' if args.tail_sensitivity else args.task
    core = ['constitutive', 'modulus', 'reference', 'conditional']
    tasks = (core + ['tail'] if selection == 'all'
             else core if selection == 'core' else [selection])
    functions = dict(constitutive=constitutive_curves,
                     modulus=modulus_scan,
                     reference=reference_distribution,
                     conditional=lambda: conditional_distribution(XI_F),
                     tail=tail_sensitivity,
                     validate=validation_only)
    metadata = dict(N=N, xi_f=XI_F, mu=mu, Delta_E=Delta_E,
                    tail_probability=TAIL, contour_margin=MARGIN,
                    tasks=tasks, completed_tasks=[], output_directory=str(OUTDIR),
                    python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, matplotlib=matplotlib.__version__)
    manifest = OUTDIR/'1_chain_run_parameters.json'
    started = time.perf_counter()
    for task in tasks:
        print(f'Running {task}; output: {OUTDIR}', flush=True)
        functions[task]()
        metadata['completed_tasks'].append(task)
        metadata['elapsed_seconds'] = time.perf_counter()-started
        manifest.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        print(f'Completed {task} ({metadata["elapsed_seconds"]:.1f} s elapsed).',
              flush=True)
    print(f'All requested tasks completed. Results: {OUTDIR}', flush=True)


if __name__ == '__main__':
    run()