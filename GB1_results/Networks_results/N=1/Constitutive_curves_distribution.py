"""
多聚蛋白网络的分布卷积本构曲线。

流程（退火 / 准静态系综）：
  1. 单域：F_d(r,n) = Lc(n) * phi(r/Lc(n)) + U(n)，Lc(n) = xi_f*(1 + n*(mu-1))。
     笛卡尔各向同性密度  u_1(r) ~ \int dn exp(-F_d(r,n))，
     支撑区间 r in [0, mu*xi_f - delta]（delta：防止不可伸长发散；该处分布已趋于零）。
  2. N 域链：串联域互相独立 =>
     u_N = u_1 (*) ... (*) u_1（3D 各向同性卷积）。
     精确简化：(g*h)(r) = (2 pi / r) \int s g(s) [H(r+s) - H(|r-s|)] ds，
     H(u) = \int_0^u t h(t) dt。均匀网格（间距 dr）=> 纯索引运算。
  3. 有效 Helmholtz 自由能 F_tilde_N(r) = -ln u_N(r) + const，
     力 f_N(r) = -d ln u_N / dr。
  4. 网络（仿射 3-chain）：
     p0(r) = 4 pi r^2 u_N(r)（已归一化）；
     sigma(lam)/(rho kBT) = (1/3) \int p0(r) r [f(lam r) - lam^-3/2 f(lam^-1/2 r)] dr
     积分限制在 lam*r 未超过轮廓长度的链上（被拉到轮廓的链不再承载；
     通过每个 N 的拉伸范围把权重保持在 1e-3 以内）。
     E0/(rho kBT) = (1/2) \int p0(r) [r^2 f'(r) + r f(r)] dr
  5. Gibbs 对比：现象学 tanh Lc(f) 单链曲线，置于固定 R0 的 3-chain 模型中，
     其中 R0 = sqrt<r^2> 由同一个 p0(r) 得到。

约化单位：kBT = lp = 1。（kBT/lp = 5.14 pN，当 lp = 0.8 nm）
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.special import logsumexp
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid

# ============ 字体与绘图风格设置（与代码 1 保持一致）============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 26
label_fontsize = 26
tick_fontsize = 24
legend_fontsize = 18
legend_title_fontsize = 20

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 2.5
lines_markersize = 12

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
xi_f     = 3.6      # 单个折叠域长度（lp）
k1_tanh  = 6.5      # Gibbs 现象学模型：tanh 斜率
k2_tanh  = 1.48     # Gibbs 现象学模型：平台力
delta    = 0.05     # 支撑区间保护：r < mu*xi_f - delta
DR       = 0.02     # 均匀网格间距（lp）
N_LIST   = [1, 2, 4, 6, 8, 16]

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def f_MS(x):
    """微观力（Marko-Siggia），kBT = lp = 1。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def U_of_n(n):
    """内部变量 n 的势能项。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

# ===================== 1. 单域分布 =====================
def single_domain_u(M=2000):
    """
    对内部变量 n 做退火，得到单域各向同性密度 u_1(r)。
    返回 (r, u_1)，均匀网格间距 DR。
    """
    Lmax1 = mu * xi_f - delta
    r = np.arange(0.0, Lmax1 + 0.5*DR, DR)          # 包含 r = 0
    n = np.linspace(0, 1, M)
    dn = n[1] - n[0]
    Lc = xi_f * (1 + n * (mu - 1))

    # x[i, j] = r_i / Lc_j
    x = np.clip(r[:, None] / Lc[None, :], 0, 1 - 1e-12)

    # 对数被积函数 ln w = -Lc * phi(x) - U(n)
    lw = -Lc[None, :] * phi(x) - U_of_n(n)[None, :]
    lnZ = logsumexp(lw, axis=1) + np.log(dn)
    lnZ[0] = lnZ[1]                                  # r=0 处光滑
    u = np.exp(lnZ - lnZ.max())
    return r, u

# ===================== 2. 各向同性 3D 卷积 =====================
def convolve_iso(g, h, dr):
    """
    对共享均匀网格（原点 0，间距 dr）上的各向同性 3D 密度做 (g*h)(r)。
    g、h 各自的支撑区间用数组长度表示。
      (g*h)(r) = (2pi/r) \int s g(s) [H(r+s) - H(|r-s|)] ds，
      H(u) = \int_0^u t h(t) dt
    均匀网格 => r_i ± s_k 都是网格点：纯索引。
      out(r_i) = (2pi/r_i) dr \sum_k s_k g_k [H_{i+k} - H_{|i-k|}]
    """
    ng, nh = len(g), len(h)
    nout = ng + nh - 1                               # 卷积支撑长度
    nH = nout + ng + 2                               # H 需要覆盖 r_i + s_max
    rg_h = np.arange(nH) * dr
    h2 = np.zeros(nH)
    h2[:nh] = h
    H = cumulative_trapezoid(rg_h * h2, rg_h, initial=0.0)

    sg = np.arange(ng) * dr * g                      # s_k * g(s_k)
    k = np.arange(ng)
    out = np.zeros(nout)

    # r = 0 极限：4 pi \int s^2 g(s) h(s) ds（共享网格上索引对齐）
    nov = min(ng, nh)
    out[0] = 4 * np.pi * dr * np.sum((np.arange(nov)*dr)**2 * g[:nov] * h[:nov])

    chunk = 4000
    for i0 in range(1, nout, chunk):
        i1 = min(i0 + chunk, nout)
        ii = np.arange(i0, i1)
        A = H[ii[:, None] + k[None, :]] - H[np.abs(ii[:, None] - k[None, :])]
        out[ii] = (2 * np.pi / (ii * dr)) * dr * (A * sg[None, :]).sum(axis=1)
    return out

def chain_density(N, u1, dr):
    """u_N = u_1 的 N 重卷积（紧支撑数组）。"""
    u = u1.copy()
    for _ in range(N - 1):
        u = convolve_iso(u, u1, dr)
        m = u.max()
        if m > 0:
            u /= m                                   # 重新缩放，常数无关
    return u

# ===================== 3. 链的可观测量 =====================
class Chain:
    """
    从 u_N(r) 网格上提取单链有效自由能、力、参考端距分布与初始模量。
    """
    def __init__(self, u, N, dr):
        self.N = N
        self.dr = dr
        rg = np.arange(len(u)) * dr

        # 截去数值零尾部
        mask = u > u.max() * 1e-270
        self.r = rg[mask]
        lnu = np.log(u[mask])

        # 用 PCHIP 平滑 log 密度，使导数稳定
        step = max(1, len(self.r) // 2000)
        knots = self.r[::step]
        self._pchip = PchipInterpolator(knots, np.interp(knots, self.r, lnu))

        self.F  = -self._pchip(self.r)                        # F_tilde（差一个常数）
        self.f  = -self._pchip.derivative(1)(self.r)          # 有效力
        self.fp = -self._pchip.derivative(2)(self.r)          # 力对 r 的导数

        # 参考态端距分布 p0(r) ~ r^2 u_N(r)，归一化
        ln_p0 = np.log(4 * np.pi) + 2 * np.log(np.maximum(self.r, 1e-12)) + lnu
        ln_p0 -= logsumexp(ln_p0 + np.log(self.r[1] - self.r[0]))
        self.p0 = np.exp(ln_p0)

        # 均方根端距
        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

    def E0(self):
        """初始模量（未断裂态）：(1/2) \int p0 [r^2 f' + r f] dr。"""
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r
        )

    def sigma(self, lam):
        """
        仿射 3-chain 名义应力。
        被拉伸到轮廓长度（lam*r 超过存储支撑，力撞到不可伸长壁）的链
        从积分中剔除：它们不再承载。用权重 wcut 平滑过渡。
        """
        out = np.zeros_like(lam)
        wtaper = 0.25              # 墙的锥度宽度（lp）：把截断铺开到约 12 个网格
        for i, lm in enumerate(lam):
            r1x = lm * self.r
            wcut = np.clip((self.r[-1] - r1x) / wtaper + 1.0, 0.0, 1.0)
            f1 = np.interp(np.minimum(r1x, self.r[-1]), self.r, self.f)
            f2 = np.interp(lm**-0.5 * self.r, self.r, self.f)
            integrand = wcut * self.p0 * self.r * (f1 - lm**-1.5 * f2)
            out[i] = trapz_func(integrand, self.r) / 3.0
        return out

    def taut_weight(self, lam):
        """在拉伸 lam 下被拉到轮廓长度的参考链权重。"""
        rc = self.r[-1] / lam
        mask = self.r >= rc
        return trapz_func(self.p0[mask], self.r[mask]) if mask.any() else 0.0

# ===================== Gibbs 现象学模型 =====================
def gibbs_sigma(N, lam, R0, x_pts=4000):
    """
    现象学 tanh 切换轮廓长度的 3-chain 模型：
        Lc(f) = N xi_f [ (mu+1)/2 + (mu-1)/2 tanh(k1 (f - k2)) ]
    """
    x = np.linspace(1e-6, 0.9995, x_pts)
    f = f_MS(x)
    Lc = N * xi_f * (0.5*(mu + 1) + 0.5*(mu - 1) * np.tanh(k1_tanh * (f - k2_tanh)))
    rr = x * Lc
    out = np.zeros_like(lam)
    for i, lm in enumerate(lam):
        f1 = np.interp(lm * R0, rr, f, right=np.nan)
        f2 = np.interp(lm**-0.5 * R0, rr, f)
        out[i] = R0 * (f1 - lm**-1.5 * f2) / 3.0
    return out

# ===================== 主程序 =====================
def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(outdir, exist_ok=True)

    print('=== 单域 ===')
    r1, u1 = single_domain_u()
    print(f'单域支撑 [0, {mu*xi_f - delta:.2f}] lp, 网格 dr = {DR}, Nr1 = {len(r1)}')

    chains = {}
    print('\n=== 卷积 & 链性质 ===')
    for N in N_LIST:
        uN = chain_density(N, u1, DR)
        ch = Chain(uN, N, DR)
        chains[N] = ch
        print(f'N={N:2d}: R0={ch.R0:7.4f}  R0^2/N={ch.R0**2/N:7.4f}  E0={ch.E0():6.3f}')

    # 矩一致性检查：R0^2 应正比于 N
    print('\n矩检验: R0^2(2) vs 2*R0^2(1):',
          f"{chains[2].R0**2:.4f} vs {2*chains[1].R0**2:.4f}")
    print('矩检验: R0^2(16) vs 16*R0^2(1):',
          f"{chains[16].R0**2:.4f} vs {16*chains[1].R0**2:.4f}")

    # 半展开处的力 vs Gibbs k2
    ch16 = chains[16]
    mid = np.argmin(np.abs(ch16.r / 16 - xi_f * (1 + mu) / 2))
    print(f'N=16 半展开伸长处力: {ch16.f[mid]:.3f}  (Gibbs k2 = {k2_tanh})')

    # ===================== 4. 绘图 =====================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # ---------------- (a) 单域参考分布 ----------------
    ax = axes[0, 0]
    pr1 = 4 * np.pi * r1**2 * u1
    pr1 /= trapz_func(pr1, r1)
    ax.plot(r1, pr1, 'b-', lw=lines_linewidth)

    # 标注折叠长度与展开长度
    ax.axvline(xi_f, color='gray', ls=':', lw=3.0)
    ax.axvline(mu * xi_f, color='gray', ls='--', lw=3.0)
    ax.text(xi_f + 0.3, 0.8, r'$\xi_f$', fontsize=25)
    ax.text(mu * xi_f - 3.5, 0.8, r'$\mu\xi_f$', fontsize=25)

    ax.set_yscale('log')
    ax.set_ylim(1e-8, 3)
    ax.set_xlabel(r'$r\ [l_p]$')
    ax.set_ylabel(r'$p_{r,1}(r)$')
    ax.set_title('Single-domain reference distribution')
    ax.grid(True, ls='--', alpha=grid_alpha)

    # ---------------- (b) 有效链力 ----------------
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(N_LIST)))
    for N, c in zip(N_LIST, colors):
        ch = chains[N]
        ax.plot(ch.r / N, ch.f, color=c, lw=lines_linewidth, label=f'$N={N}$')

    # Gibbs 现象学曲线（黑虚线）
    xg = np.linspace(1e-6, 0.999, 3000)
    fg = f_MS(xg)
    Lc_g = xi_f * (0.5*(mu + 1) + 0.5*(mu - 1) * np.tanh(k1_tanh * (fg - k2_tanh)))
    ax.plot(xg * Lc_g, fg, 'k--', lw=lines_linewidth, label='Gibbs tanh')

    ax.set_xlabel(r'$r/N\ [l_p]$')
    ax.set_ylabel(r'$\tilde f_N(r)\ [k_BT/l_p]$')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 5)
    ax.set_title('Effective chain force')
    ax.legend(fontsize=legend_fontsize*0.7, loc='best', framealpha=0.8, edgecolor='none')
    ax.grid(True, ls='--', alpha=grid_alpha)

    # ---------------- (c) 网络本构曲线 ----------------
    ax = axes[1, 0]
    # 每个 N 的拉伸范围：拉伸到同一轮廓比例 ETA，
    # lam_max = ETA * N*mu*xi_f / R0（随 sqrt(N) 增长），
    # 且在被拉到轮廓的参考链权重超过 WTOL 之前停止。
    # WTOL 取足够大，使曲线在上端明显进入有限伸长硬化区。
    ETA, WTOL = 0.55, 5e-3
    for N, c in zip(N_LIST, colors):
        ch = chains[N]
        hi = ETA * N * mu * xi_f / ch.R0
        lo = 1.0
        if ch.taut_weight(hi) > WTOL:          # 二分：满足 W <= WTOL 的最大 lam
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if ch.taut_weight(mid) <= WTOL:
                    lo = mid
                else:
                    hi = mid
            hi = lo

        lam = np.linspace(1.0, hi, 160)
        sig = ch.sigma(lam)
        sig_g = gibbs_sigma(N, lam, ch.R0)

        ax.plot(lam, sig, color=c, lw=lines_linewidth,
                label=f'$N={N}$ ')
        ax.plot(lam, sig_g, color=c, lw=1.2, ls='--', alpha=0.75,
                label=f'$N={N}$ Gibbs 3-chain')
        ax.plot(lam[-1], sig[-1], 'o', color=c, ms=6)

        print(f'N={N:2d}: lambda_max = {hi:6.2f}  taut weight = {ch.taut_weight(hi):.2e}'
              f'  sigma(lambda_max) = {sig[-1]:8.3f}')

    ax.set_xlim(1.0, 20.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel(r'Stretch $\lambda$')
    ax.set_ylabel(r'$\sigma/(\rho k_BT)$')
    ax.set_title('Network constitutive curves')
    ax.legend(fontsize=10, ncol=2, framealpha=0.9, edgecolor='none')
    ax.grid(True, ls='--', alpha=grid_alpha)

    # ---------------- (d) 初始模量 vs 域数 ----------------
    ax = axes[1, 1]
    E0s = [chains[N].E0() for N in N_LIST]
    ax.plot(N_LIST, E0s, 'o-', color='navy', lw=lines_linewidth, ms=8,
            label='annealed')
    ax.axhline(3.0, color='red', ls='--', lw=2, label=r'$G_0^{Rubber}=3\rho k_BT$')
    # ax.axhline(6.46, color='gray', ls=':', lw=2, label='N=1 frozen-n (high-freq)')

    for N, E in zip(N_LIST, E0s):
        ax.annotate(f'{E:.2f}', (N, E), textcoords='offset points',
                    xytext=(8, 6), fontsize=11)

    ax.set_xlim(1.0, 16.0)
    ax.set_ylim(2.0, 5.0)
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$G_0/(\rho k_BT)$')
    ax.set_title('Initial modulus')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, ls='--', alpha=grid_alpha)
    ax.set_xscale('log')
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST])

    fig.tight_layout()
    fpath = os.path.join(outdir, 'network_convolution_scheme.png')
    fig.savefig(fpath, dpi=savefig_dpi, facecolor='white', edgecolor='none')
    print('\nsaved:', fpath)


if __name__ == '__main__':
    main()