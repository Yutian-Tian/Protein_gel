"""
================================================================================
 N = 1 可解折叠网络的严格本构曲线（可视化与代码 2 对齐版）
 + 经典 WLC 网络基线的两条本构曲线（依据《经典 WLC 网络方案报告》）：
     · 完全折叠    L = ξf
     · 完全解折叠  L = μ·ξf
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.special import logsumexp
from scipy.interpolate import PchipInterpolator

# ============ 字体与绘图风格设置（与代码 2 完全对齐） ============
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

trapz_func = getattr(np, 'trapezoid', np.trapz)

# ============ 核心物理参数 ============
mu       = 7.6      # 展开态/折叠态轮廓长度比
Delta_E  = 11.7     # 内部态能量尺度（kBT）
xi_f     = 3.6      # 折叠态单域长度（lp）
delta    = 0.05     # 支撑区间保护
DR       = 0.02     # 均匀网格间距（lp）

# 数值稳定参数
F_MAX_CLIP  = 1.0e4
FP_MAX_CLIP = 1.0e4
U_TAIL      = 1.0e-300

WTOL = 5e-3

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def U_of_n(n):
    """内部变量 n 的势能项。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

def fMS(x):
    """Marko–Siggia 无量纲链力"""
    x = np.clip(x, 0.0, 1.0 - 1e-9)
    return 0.25 / (1.0 - x)**2 - 0.25 + x

def fMSp(x):
    """df_MS/dx"""
    x = np.clip(x, 0.0, 1.0 - 1e-9)
    return 0.5 / (1.0 - x)**3 + 1.0

# ===================== 单域退火分布 u(r) =====================
def single_domain_u(M=2000):
    Lmax1 = mu * xi_f - delta
    r = np.arange(0.0, Lmax1 + 0.5*DR, DR)
    n = np.linspace(0, 1, M)
    dn = n[1] - n[0]
    Lc = xi_f * (1 + n * (mu - 1))

    x = np.clip(r[:, None] / Lc[None, :], 0, 1 - 1e-12)
    lw = -Lc[None, :] * phi(x) - U_of_n(n)[None, :]
    lnZ = logsumexp(lw, axis=1) + np.log(dn)
    lnZ[0] = lnZ[1]                       # r = 0 处光滑
    u = np.exp(lnZ - lnZ.max())
    return r, u

# ===================== N = 1 链的可观测量 =====================
class ChainN1:
    def __init__(self, r, u):
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.clip(u, 0.0, None)

        mask = u > max(u.max() * U_TAIL, 1e-300)
        if mask.sum() < 4:
            mask = u > 0
        self.r = r[mask]
        self.u = u[mask]
        lnu = np.log(self.u)

        step = max(1, len(self.r) // 2000)
        knots = self.r[::step]
        if knots[-1] < self.r[-1]:
            knots = np.append(knots, self.r[-1])
        self._pchip = PchipInterpolator(knots, np.interp(knots, self.r, lnu))

        f_raw  = -self._pchip.derivative(1)(self.r)
        fp_raw = -self._pchip.derivative(2)(self.r)

        self.f  = np.clip(np.nan_to_num(f_raw,  nan=0.0, posinf=0.0, neginf=0.0),
                          0.0, F_MAX_CLIP)
        self.fp = np.clip(np.nan_to_num(fp_raw, nan=0.0, posinf=0.0, neginf=0.0),
                          0.0, FP_MAX_CLIP)

        if len(self.f) > 0:
            self.f[0] = 0.0
            if len(self.fp) > 1:
                self.fp[0] = self.fp[1]

        ln_p0 = np.log(4 * np.pi) + 2 * np.log(np.maximum(self.r, 1e-12)) + lnu
        ln_p0 -= logsumexp(ln_p0 + np.log(self.r[1] - self.r[0]))
        self.p0 = np.exp(ln_p0)

        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

    def sigma(self, lam):
        out = np.zeros_like(lam)
        wtaper = 0.25
        for i, lm in enumerate(lam):
            r1x = lm * self.r
            wcut = np.clip((self.r[-1] - r1x) / wtaper + 1.0, 0.0, 1.0)
            f1 = np.interp(np.minimum(r1x, self.r[-1]), self.r, self.f)
            f2 = np.interp(lm**-0.5 * self.r, self.r, self.f)
            integrand = wcut * self.p0 * self.r * (f1 - lm**-1.5 * f2)
            out[i] = trapz_func(integrand, self.r) / 3.0
        return out

    def E0(self):
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r
        )

    def taut_weight(self, lam):
        rc = self.r[-1] / lam
        mask = self.r >= rc
        return trapz_func(self.p0[mask], self.r[mask]) if mask.any() else 0.0

# ===================== 经典 WLC 网络 =====================
class WLCNetwork:
    def __init__(self, L, delta=delta, dr=DR):
        self.L = float(L)
        self.delta = delta
        r = np.arange(0.0, self.L - delta + 0.5 * dr, dr)
        x = r / self.L

        F = self.L * phi(x)
        ln_p0 = np.log(4.0 * np.pi) + 2.0 * np.log(np.maximum(r, 1e-12)) - F
        ln_p0 -= logsumexp(ln_p0 + np.log(r[1] - r[0]))
        self.p0 = np.exp(ln_p0)
        self.r = r

        self.f  = np.clip(fMS(x), 0.0, F_MAX_CLIP)
        self.fp = np.clip(fMSp(x) / self.L, 0.0, FP_MAX_CLIP)

        self.R0 = np.sqrt(trapz_func(r**2 * self.p0, r))

    def sigma(self, lam):
        out = np.zeros_like(lam)
        wtaper = 0.25
        for i, lm in enumerate(lam):
            r1x = lm * self.r
            wcut = np.clip((self.r[-1] - r1x) / wtaper + 1.0, 0.0, 1.0)
            f1 = np.interp(np.minimum(r1x, self.r[-1]), self.r, self.f)
            f2 = np.interp(lm**-0.5 * self.r, self.r, self.f)
            integrand = wcut * self.p0 * self.r * (f1 - lm**-1.5 * f2)
            out[i] = trapz_func(integrand, self.r) / 3.0
        return out

    def E0(self):
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r
        )

    def taut_weight(self, lam):
        rc = self.r[-1] / lam
        mask = self.r >= rc
        return trapz_func(self.p0[mask], self.r[mask]) if mask.any() else 0.0

def lam_upper_bound(net, Lwall, wtol=WTOL):
    hi = 0.98 * Lwall / net.R0
    lo = 1.0
    if hi <= lo:
        hi = 1.0 + 1e-9
    if net.taut_weight(hi) > wtol:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if net.taut_weight(mid) <= wtol:
                lo = mid
            else:
                hi = mid
        hi = lo
    return hi

# ===================== 主程序 =====================
def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(outdir, exist_ok=True)

    # ---------- N = 1 单域退火 ----------
    print('=== N = 1 单域退火 ===')
    r, u = single_domain_u()
    ch = ChainN1(r, u)
    print(f'支撑区间 [0, {ch.r[-1]:.3f}] lp，有效网格点数 {len(ch.r)}')
    print(f'R0 = {ch.R0:.4f} lp')
    print(f'E0 = {ch.E0():.4f} ρkBT')

    ETA = 0.55
    hi = ETA * 1 * mu * xi_f / ch.R0
    lo = 1.0
    if ch.taut_weight(hi) > WTOL:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ch.taut_weight(mid) <= WTOL:
                lo = mid
            else:
                hi = mid
        hi = lo

    lam = np.linspace(1.0, hi, 160)
    sig = ch.sigma(lam)
    print(f'lambda_max = {hi:.2f}, 末端应力 = {sig[-1]:.3f}')

    # ---------- 经典 WLC 网络基线 ----------
    print('\n=== 经典 WLC 网络基线（3-chain 仿射，分布版） ===')
    wlc_folded   = WLCNetwork(xi_f)        
    wlc_unfolded = WLCNetwork(mu * xi_f)   

    E0_f = wlc_folded.E0()
    E0_u = wlc_unfolded.E0()

    lam_f = np.linspace(1.0, lam_upper_bound(wlc_folded, wlc_folded.L), 160)
    sig_f = wlc_folded.sigma(lam_f)
    print(f'[WLC folded]   L = {wlc_folded.L:.2f} lp, R0 = {wlc_folded.R0:.4f} lp, '
          f'E0 = {E0_f:.4f} ρkBT, lambda_max = {lam_f[-1]:.2f}')

    lam_u = np.linspace(1.0, lam_upper_bound(wlc_unfolded, wlc_unfolded.L), 160)
    sig_u = wlc_unfolded.sigma(lam_u)
    print(f'[WLC unfolded] L = {wlc_unfolded.L:.2f} lp, R0 = {wlc_unfolded.R0:.4f} lp, '
          f'E0 = {E0_u:.4f} ρkBT, lambda_max = {lam_u[-1]:.2f}')

    # ---------- 绘图：与代码 2 风格完全对齐 ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0, 5.0)

    # 画线，统一使用 LaTeX 数学环境
    ax.plot(lam, sig, 'r-', linewidth=lines_linewidth, label=r'$N=1$ protein network')
    #ax.plot(lam_f, sig_f, 'b--', linewidth=lines_linewidth, label=fr'WLC folded, \(L=\xi_f\) (\(E_0={E0_f:.2f}\ \rho k_B T\))')
    #ax.plot(lam_u, sig_u, 'g-.', linewidth=lines_linewidth, label=fr'WLC unfolded, \(L=\mu\xi_f\) (\(E_0={E0_u:.2f}\ \rho k_B T\))')

    # 坐标轴标签与标题，统一使用 LaTeX 数学环境
    ax.set_xlabel(r'Stretch $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'Stress $\sigma / \rho k_B T$', fontsize=label_fontsize)
    ax.set_title('Constitutive curve', fontsize=title_fontsize, pad=20)

    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    # ---- major tick ----
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    
    # ---- minor tick ----
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, width=1,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=4, width=1,
                   direction=ytick_direction, right=ytick_right)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    ax.legend(fontsize=0.8*legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    plt.tight_layout()
    fpath = os.path.join(outdir, 'Constitutive_Curve_N1.png')
    fig.savefig(fpath, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('\nsaved:', fpath)

if __name__ == '__main__':
    main()