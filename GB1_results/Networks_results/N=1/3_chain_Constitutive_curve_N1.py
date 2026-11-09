"""
================================================================================
 N = 1 四模型本构比较：原蛋白网络、两种WLC、第17页模型(gibbs network)
 推导见 derivations/gibbs_constitutive_N1.md
 + 内部态直接积分，参考态尾部在变形前固定截断
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
from scipy.integrate import cumulative_trapezoid

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

trapz_func = np.trapezoid

# ============ 核心物理参数 ============
mu       = 7.6      # 展开态/折叠态轮廓长度比
Delta_E  = 11.9     # 内部态能量尺度（kBT）
xi_f     = 3.6      # 折叠态单域长度（lp）
delta    = 0.05     # 支撑区间保护
DR       = 0.005     # 均匀网格间距（lp）

# 有限参考系综：在变形前固定去除最外侧 WTOL 概率并重新归一化。
# 此近似使保留链都满足不可伸长约束；不能解释为完整平衡系综的极限。
WTOL = 5e-3
CONTOUR_MARGIN = 0.995
N_QUAD = 512

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

# ===================== 单域退火统计 =====================
def domain_statistics(r, M=N_QUAD):
    """直接积分内部态：返回 log Z、有效力和有效刚度，不封顶或裁掉负刚度。

    f_eff = <f_n>，f_eff' = <f_n'> - Var(f_n)，kBT = lp = 1。
    每个端距将 n 的积分区间限定为 Lc(n) > r。
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0) or np.any(r >= mu * xi_f):
        raise ValueError('Protein end distance must satisfy 0 <= r < mu*xi_f')
    q, w = np.polynomial.legendre.leggauss(M)
    lower = np.maximum(0.0, (r / xi_f - 1.0) / (mu - 1.0))
    half = 0.5 * (1.0 - lower)
    n = lower[..., None] + half[..., None] * (q + 1.0)
    Lc = xi_f * (1.0 + (mu - 1.0) * n)
    x = r[..., None] / Lc
    lw = -Lc * phi(x) - U_of_n(n) + np.log(half[..., None] * w)
    logZ = logsumexp(lw, axis=-1)
    prob = np.exp(lw - logZ[..., None])
    force = fMS(x)
    f = np.sum(prob * force, axis=-1)
    fp = np.sum(prob * fMSp(x) / Lc, axis=-1) - np.sum(
        prob * (force - f[..., None])**2, axis=-1)
    return logZ, f, fp


def single_domain_u(M=N_QUAD):
    r = np.arange(0.0, mu * xi_f - delta, DR)
    logZ, _, _ = domain_statistics(r, M)
    return r, np.exp(logZ - logZ.max())


def finite_reference(r, p0, tail=WTOL):
    """变形前固定参考分布；截去 tail 尾概率，绝不随 lambda 删链。"""
    cdf = cumulative_trapezoid(p0, r, initial=0.0)
    cdf /= cdf[-1]
    cutoff = np.interp(1.0 - tail, cdf, r)
    keep = r < cutoff
    rg = np.append(r[keep], cutoff)
    pg = np.interp(rg, r, p0)
    pg /= trapz_func(pg, rg)
    return rg, pg


class ChainN1:
    def __init__(self, r, u, tail=WTOL):
        p0 = r**2 * u
        self.r, self.p0 = finite_reference(r, p0, tail)
        self.L = mu * xi_f
        self.f, self.fp = domain_statistics(self.r)[1:]
        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

    def sigma(self, lam):
        out = np.zeros_like(lam, dtype=float)
        for i, lm in enumerate(lam):
            f1 = domain_statistics(lm * self.r)[1]
            f2 = domain_statistics(lm**-0.5 * self.r)[1]
            out[i] = trapz_func(self.p0 * self.r *
                               (f1 - lm**-1.5 * f2), self.r) / 3.0
        return out

    def E0(self):
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r)


class WLCNetwork:
    """固定轮廓 WLC；解折叠基线继承蛋白网络的固定参考分布。"""
    def __init__(self, L, delta=delta, dr=DR, reference=None, tail=WTOL):
        self.L = float(L)
        self._own_ref = reference is None
        if reference is None:
            r = np.arange(0.0, self.L - delta, dr)
            p0 = r**2 * np.exp(-self.L * phi(r / self.L))
            self.r, self.p0 = finite_reference(r, p0, tail)
        else:
            self.r, self.p0 = (np.asarray(v, dtype=float) for v in reference)
        self.f = self.force(self.r)
        self.fp = fMSp(self.r / self.L) / self.L
        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

    def force(self, r):
        if np.any(np.asarray(r) >= self.L):
            raise ValueError('WLC end distance must remain below its contour length')
        return fMS(np.asarray(r) / self.L)

    def sigma(self, lam):
        return np.array([trapz_func(self.p0 * self.r *
                         (self.force(lm * self.r) - lm**-1.5 *
                          self.force(lm**-0.5 * self.r)), self.r) / 3.0
                         for lm in lam])

    def E0(self):
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r)


def lam_upper_bound(net, Lwall):
    """所有保留参考链均在轮廓以内；相同参考态及轮廓有相同上限。"""
    return CONTOUR_MARGIN * Lwall / net.r[-1]

# ===================== 第17页：Gibbs参数化单链 =====================
K1 = 6.5
K2 = 1.48


class GibbsNetwork:
    """参考态继承原蛋白网络，仅将变形自由能换为积分 f_G 得到的 F_G。适用Gibbs力拉伸曲线"""
    def __init__(self, reference, N=1):
        self.r, self.p0 = reference
        self.N = N
        self.L = N * mu * xi_f

    def contour(self, f):
        t = np.tanh(K1 * (f - K2))
        L = self.N * xi_f * ((mu + 1) / 2 + (mu - 1) * t / 2)
        Lp = self.N * xi_f * (mu - 1) * K1 * (1 - t*t) / 2
        return L, Lp

    def force_stiffness(self, r):
        r = np.asarray(r, dtype=float)
        if np.any(r < 0) or np.any(r >= self.L):
            raise ValueError('gibbs end distance must satisfy 0 <= r < Lmax')
        lo = np.zeros_like(r)
        hi = np.full_like(r, 1.0 - 1e-10)
        for _ in range(60):
            x = (lo + hi) / 2
            distance = x * self.contour(fMS(x))[0]
            below = distance < r
            lo = np.where(below, x, lo)
            hi = np.where(below, hi, x)
        x = (lo + hi) / 2
        f = fMS(x)
        L, Lp = self.contour(f)
        fp = fMSp(x) / (L + x * Lp * fMSp(x))
        return f, fp

    def sigma(self, lam):
        return np.array([trapz_func(self.p0 * self.r *
            (self.force_stiffness(lm * self.r)[0] - lm**-1.5 *
             self.force_stiffness(lm**-0.5 * self.r)[0]), self.r) / 3
             for lm in lam])

    def E0(self):
        f, fp = self.force_stiffness(self.r)
        return trapz_func(self.p0 * (self.r**2 * fp + self.r * f), self.r) / 2

    def chain_energy(self, r, order=512):
        """F_G(r)=integral_0^r f_G(s)ds，独立于网络应力积分。"""
        r = np.asarray(r)
        q, w = np.polynomial.legendre.leggauss(order)
        t = (q + 1) / 2
        return r * np.sum(self.force_stiffness(r[..., None] * t)[0] * w / 2,
                          axis=-1)

    def network_energy(self, lam, order=512):
        return trapz_func(self.p0 * (self.chain_energy(lam * self.r, order) +
            2 * self.chain_energy(lam**-0.5 * self.r, order)), self.r) / 3


def validate_gibbs(net):
    """验证反演、初始模量以及自由能导数；返回验证文本行。"""
    distances = np.linspace(0, 0.995 * net.L, 300)
    force, _ = net.force_stiffness(distances)
    # 从 f 独立反演 Marko-Siggia x，检查 r(f)。
    lo, hi = np.zeros_like(force), np.ones_like(force)
    for _ in range(60):
        x = (lo + hi) / 2
        below = fMS(x) < force
        lo = np.where(below, x, lo)
        hi = np.where(below, hi, x)
    residual = np.max(np.abs((lo + hi) / 2 * net.contour(force)[0] - distances))
    assert residual < 1e-10
    h = 1e-4
    slope = (net.sigma([1+h])[0] - net.sigma([1-h])[0]) / (2*h)
    assert np.isclose(slope, net.E0(), rtol=1e-6)
    assert abs(net.sigma([1.0])[0]) < 1e-12
    lines = [f'force inversion max residual: {residual:.8g}',
             f'E0 analytic: {net.E0():.10g}; finite difference: {slope:.10g}']
    for lm in [2.0, 6.0]:
        derivative = (net.network_energy(lm+h) - net.network_energy(lm-h)) / (2*h)
        stress = net.sigma([lm])[0]
        energy256 = net.network_energy(lm, 256)
        energy512 = net.network_energy(lm, 512)
        assert np.isclose(derivative, stress, rtol=2e-4, atol=1e-7)
        assert np.isclose(energy256, energy512, rtol=2e-5, atol=1e-7)
        lines.append(f'lambda={lm:g}: dF/dlambda={derivative:.10g}, '
                     f'sigma={stress:.10g}, F(256)={energy256:.10g}, '
                     f'F(512)={energy512:.10g}')
    return lines


# ===================== 计算 + 验证 + 落盘 + 绘图 =====================
def plot_constitutive_curve(outdir, N=1):
    """计算并绘制 N 段蛋白网络、两种 WLC 基线与 Gibbs 网络的本构曲线。

    函数只接收输出目录与链段数；所有物理计算、验证、CSV/验证文本落盘
    以及绘图都在函数内完成。返回 dict 便于调用方复用计算结果。
    """
    # ---------- N 段单域退火 ----------
    print(f'=== N = {N} 单域退火 ===')
    r, u = single_domain_u()
    ch = ChainN1(r, u)
    print(f'支撑区间 [0, {ch.r[-1]:.3f}] lp，有效网格点数 {len(ch.r)}')
    print(f'R0 = {ch.R0:.4f} lp')
    print(f'E0 = {ch.E0():.4f} ρkBT')

    print(f'Fixed reference tail probability removed: {WTOL:g}')
    hi = lam_upper_bound(ch, ch.L)
    lam = np.linspace(1.0, hi, 160)
    sig = ch.sigma(lam)
    print(f'lambda_max = {hi:.2f}, 末端应力 = {sig[-1]:.3f}')

    # ---------- 经典 WLC 网络基线 ----------
    print('\n=== 经典 WLC 网络基线（3-chain 仿射，分布版） ===')
    wlc_folded   = WLCNetwork(xi_f)
    wlc_unfolded = WLCNetwork(mu * xi_f, reference=(ch.r, ch.p0))

    E0_f = wlc_folded.E0()
    E0_u = wlc_unfolded.E0()

    lam_f = np.linspace(1.0, lam_upper_bound(wlc_folded, wlc_folded.L), 160)
    sig_f = wlc_folded.sigma(lam_f)
    print(f'[WLC folded]   L = {wlc_folded.L:.2f} lp, R0 = {wlc_folded.R0:.4f} lp, '
          f'E0 = {E0_f:.4f} ρkBT, lambda_max = {lam_f[-1]:.2f}')

    lam_u = lam.copy()  # 同一参考分布、同一最大轮廓长度、同一有效伸长区间
    sig_u = wlc_unfolded.sigma(lam_u)
    print(f'[WLC unfolded] L = {wlc_unfolded.L:.2f} lp, R0 = {wlc_unfolded.R0:.4f} lp, '
          f'E0 = {E0_u:.4f} ρkBT, lambda_max = {lam_u[-1]:.2f}')

    # ---------- Gibbs 网络模型 ----------
    gibbsnetwork = GibbsNetwork((ch.r, ch.p0), N=N)
    sig_g = gibbsnetwork.sigma(lam)
    print(f'[Gibbs] k1={K1}, k2={K2}, E0={gibbsnetwork.E0():.6f} rho*kBT')
    validation = validate_gibbs(gibbsnetwork)
    for line in validation:
        print(line)
    with open(os.path.join(outdir, f'Gibbs_N{N}_validation.txt'), 'w') as report:
        report.write(f'N={N}, xi_f={xi_f}, mu={mu}, Delta_E={Delta_E}, '
                     f'k1={K1}, k2={K2}, tail={WTOL}, DR={DR}\n')
        report.write('\n'.join(validation) + '\n')

    # ---------- 顺序检查与 CSV 落盘 ----------
    gap = sig - sig_u
    print(f'min(sigma_protein - sigma_unfolded) = {gap.min():.8g}')
    if gap.min() < -1e-8:
        raise RuntimeError('Protein/WLC stress ordering failed; '
                           'inspect model and quadrature')

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

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(0, 5.0)

    ax.plot(lam, sig, 'r-', linewidth=lines_linewidth,
            label=rf'$N={N}$ protein network')
    ax.plot(lam_f, sig_f, 'b--', linewidth=lines_linewidth,
            label=r'WLC folded')
    ax.plot(lam_u, sig_u, 'g--', linewidth=lines_linewidth,
            label=r'WLC unfolded')
    ax.plot(lam, sig_g, 'k-.', linewidth=lines_linewidth,
            label=r'Gibbs network model')

    ax.set_xlabel(r'Stretch $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'Stress $\sigma / \rho k_B T$', fontsize=label_fontsize)
    ax.set_title('Constitutive curve', fontsize=title_fontsize, pad=20)

    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    # ---- major tick ----
    ax.tick_params(axis='x', which='major', length=6,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width,
                   direction=ytick_direction, right=ytick_right)
    # ---- minor tick ----
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, width=1,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=4, width=1,
                   direction=ytick_direction, right=ytick_right)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    plt.tight_layout()
    fpath = os.path.join(outdir, f'Constitutive_Curve_N{N}.png')
    fig.savefig(fpath, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('saved:', fpath)

    return dict(lam=lam, sig=sig, lam_f=lam_f, sig_f=sig_f,
                lam_u=lam_u, sig_u=sig_u, sig_g=sig_g,
                chain=ch, wlc_folded=wlc_folded,
                wlc_unfolded=wlc_unfolded, gibbs=gibbsnetwork,
                validation=validation)


# ===================== 主程序 =====================
def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(outdir, exist_ok=True)
    plot_constitutive_curve(outdir, N=1)


if __name__ == '__main__':
    main()