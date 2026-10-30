# -*- coding: utf-8 -*-
"""
单链（N=1）有效力曲线 + 两条 WLC 力拉伸边界。

来源：Single_chain_N1.py 子图(b)"Effective chain force"中 N=1 曲线的独立计算。
边界：
  - WLC folded,   L = xi_f    ：完全折叠轮廓长度下的 Marko-Siggia 力
  - WLC unfolded, L = mu*xi_f ：完全解折叠轮廓长度下的 Marko-Siggia 力

约化单位：kBT = lp = 1。
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

# ============ 字体与绘图风格设置（与 Constitutive_curves1.py 保持一致）============
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

trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ============ 核心物理参数 ============
mu       = 7.6      # 展开态/折叠态轮廓长度比
Delta_E  = 11.9     # 内部态能量尺度（kBT），单分散基准
xi_f     = 3.6      # 单个折叠域长度（lp）
delta    = 0.05     # 支撑区间保护：r < mu*xi_f - delta
DR       = 0.02     # 均匀网格间距（lp）

# 多分散能垒采样参数
DE_MEAN  = 11.9     # 高斯均值（kBT）
DE_STD   = 1.7      # 高斯标准差（kBT）
DE_HW    = 5.0      # 截断半宽：Delta_E ∈ [DE_MEAN-DE_HW, DE_MEAN+DE_HW]
N_SAMPLE = 100      # 采样次数
SEED     = 42

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def f_MS(x):
    """微观力（Marko-Siggia），kBT = lp = 1。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def U_of_n(n, DE=Delta_E):
    """内部变量 n 的势能项。"""
    return DE * n - DE * np.cos(2 * np.pi * n)

# ===================== 1. 单域分布（N=1 链密度） =====================
def single_domain_u(M=2000, DE=Delta_E):
    """
    对内部变量 n 做退火，得到单域各向同性密度 u_1(r; Delta_E)。
    返回 (r, u_1)，均匀网格间距 DR。
    """
    Lmax1 = mu * xi_f - delta
    r = np.arange(0.0, Lmax1 + 0.5*DR, DR)
    n = np.linspace(0, 1, M)
    dn = n[1] - n[0]
    Lc = xi_f * (1 + n * (mu - 1))

    x = np.clip(r[:, None] / Lc[None, :], 0, 1 - 1e-12)

    lw = -Lc[None, :] * phi(x) - U_of_n(n, DE)[None, :]
    lnZ = logsumexp(lw, axis=1) + np.log(dn)
    lnZ[0] = lnZ[1]
    u = np.exp(lnZ - lnZ.max())
    return r, u

# ===================== 2. 链的可观测量（取 N=1） =====================
class Chain:
    """从 u_N(r) 网格上提取单链有效自由能、力、参考端距分布与初始模量。"""
    def __init__(self, u, dr):
        self.dr = dr
        rg = np.arange(len(u)) * dr

        mask = u > u.max() * 1e-270
        self.r = rg[mask]
        lnu = np.log(u[mask])

        step = max(1, len(self.r) // 2000)
        knots = self.r[::step]
        self._pchip = PchipInterpolator(knots, np.interp(knots, self.r, lnu))

        self.F  = -self._pchip(self.r)
        self.f  = -self._pchip.derivative(1)(self.r)
        self.fp = -self._pchip.derivative(2)(self.r)

        ln_p0 = np.log(4 * np.pi) + 2 * np.log(np.maximum(self.r, 1e-12)) + lnu
        ln_p0 -= logsumexp(ln_p0 + np.log(self.r[1] - self.r[0]))
        self.p0 = np.exp(ln_p0)

        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

# ===================== 多分散能垒系综 =====================
def sample_delta_E(n_samples=N_SAMPLE, mean=DE_MEAN, std=DE_STD,
                   half_width=DE_HW, seed=SEED):
    """从截断高斯 N(mean, std^2) ∩ [mean-hw, mean+hw] 采样能垒。"""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_samples:
        cand = rng.normal(mean, std, size=n_samples * 4)
        cand = cand[(cand >= mean - half_width) & (cand <= mean + half_width)]
        out.extend(cand.tolist())
    return np.array(out[:n_samples])

def polydisperse_force(r_grid, DEs):
    """
    对每个采样能垒计算单链有效力曲线，对齐到公共网格 r_grid。
    返回 F_all（形状 [n_samples, len(r_grid)]）。
    """
    F_all = np.empty((len(DEs), len(r_grid)))
    for k, DE in enumerate(DEs):
        _, u = single_domain_u(DE=DE)
        ch = Chain(u, DR)
        F_all[k] = np.interp(r_grid, ch.r, np.clip(ch.f, 0.0, 1e4))
    return F_all

# ===================== Maxwell 等面积构造验证 =====================
def _crossings(r, y):
    """线性插值求 y(r) 的零点（符号变号）。"""
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    roots = [r[i] - y[i] * (r[i+1] - r[i]) / (y[i+1] - y[i]) for i in idx]
    return np.array(roots)

def maxwell_check(r, f, rg, fg, k2):
    """
    验证：黑色 Gibbs 曲线是否近似为红色 f(r) 的 Maxwell 构造。

    步骤：
      1. 找红色曲线的峰与（峰后）谷；
      2. 对水平线 f*：取上升支第一个交点 r1、下降支交点 r2、恢复支交点 r3，
         定义净面积 A(f*) = ∫_{r1}^{r3} [f(r) - f*] dr，
         二分求 A(f*) = 0 的 f*_Maxwell（峰谷间上方/下方包裹面积相等）；
      3. 在 f* = k2（Gibbs 平台力）处计算同样的两块包裹面积，
         以及与黑色曲线本身的面积差，判断近似程度。
    """
    # 峰/谷限定在解折叠过渡区（r < 12 lp），避开末端轮廓硬化区
    m_win = r < 12.0
    i_pk = int(np.argmax(np.where(m_win, f, -np.inf)))
    i_vy = i_pk + int(np.argmin(f[i_pk:]))
    f_pk, f_vy = f[i_pk], f[i_vy]
    r_pk, r_vy = r[i_pk], r[i_vy]
    print(f'[Maxwell] 峰: r={r_pk:.3f}, f={f_pk:.3f} | 谷: r={r_vy:.3f}, f={f_vy:.3f}')

    def lobes(f_star):
        """返回 (r1, r2, r3, A_above, A_below)；交点不足时返回 None。"""
        roots = _crossings(r, f - f_star)
        pre  = roots[roots < r_pk]          # 上升支交点
        mid  = roots[(roots > r_pk) & (roots < r_vy)]  # 峰谷间（下降支）
        post = roots[roots > r_vy]          # 恢复支交点
        if len(pre) == 0 or len(mid) == 0 or len(post) == 0:
            return None
        r1, r2, r3 = pre[-1], mid[0], post[0]
        m13 = (r >= r1) & (r <= r3)
        m12 = (r >= r1) & (r <= r2)
        m23 = (r >= r2) & (r <= r3)
        A_above = trapz_func(np.maximum(f[m12] - f_star, 0.0), r[m12])
        A_below = trapz_func(np.maximum(f_star - f[m23], 0.0), r[m23])
        A_net   = trapz_func(f[m13] - f_star, r[m13])
        return r1, r2, r3, A_above, A_below, A_net

    # ---- 二分求 f*_Maxwell ----
    lo, hi = f_vy + 1e-6, f_pk - 1e-6
    for _ in range(80):
        mid_f = 0.5 * (lo + hi)
        res = lobes(mid_f)
        if res is None:
            break
        if res[-1] > 0:      # 净面积为正 -> f* 偏低
            lo = mid_f
        else:
            hi = mid_f
    f_maxwell = 0.5 * (lo + hi)
    res = lobes(f_maxwell)
    r1, r2, r3, A_ab, A_bl, A_net = res
    print(f'[Maxwell] f*_Maxwell = {f_maxwell:.4f} kBT/lp')
    print(f'[Maxwell] 交点: r1={r1:.3f}, r2={r2:.3f}, r3={r3:.3f}')
    print(f'[Maxwell] 上方包裹面积 A_above = {A_ab:.4f}, 下方 A_below = {A_bl:.4f}, '
          f'净残差 = {A_net:.2e}')

    # ---- f* = k2（Gibbs 平台）处的包裹面积 ----
    res_k2 = lobes(k2)
    if res_k2 is not None:
        r1k, r2k, r3k, A_ab_k2, A_bl_k2, A_net_k2 = res_k2
        print(f'[Maxwell] 取 f*=k2={k2}: 交点 r1={r1k:.3f}, r2={r2k:.3f}, r3={r3k:.3f}')
        print(f'[Maxwell] 上方包裹面积 = {A_ab_k2:.4f}, 下方 = {A_bl_k2:.4f}, '
              f'净面积 = {A_net_k2:+.4f} (kBT/lp)^2? 单位 kBT')

    # ---- 与黑色 Gibbs 曲线的直接面积比较（在 [r1, r3] 上）----
    m13 = (r >= r1) & (r <= r3)
    f_black = np.interp(r[m13], rg, fg)
    A_diff = trapz_func(f[m13] - f_black, r[m13])
    A_abs  = trapz_func(np.abs(f[m13] - f_black), r[m13])
    print(f'[Maxwell] [r1,r3] 内 红线-黑线 净面积 = {A_diff:+.4f}, 绝对面积 = {A_abs:.4f}')
    print(f'[Maxwell] k2 - f*_Maxwell = {k2 - f_maxwell:+.4f}')

    return dict(f_maxwell=f_maxwell, r1=r1, r2=r2, r3=r3,
                A_above=A_ab, A_below=A_bl, res_k2=res_k2)


# ===================== Maxwell 构造示意图（通用） =====================
def plot_maxwell_figure(r, f, rg, fg, k2, title, fpath, xlim=(0.0, 20.0)):
    """对给定力曲线做 Maxwell 验证并绘制示意图，返回验证结果字典。"""
    mc = maxwell_check(r, f, rg, fg, k2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, f, 'r-', linewidth=lines_linewidth,
            label=r'force curve')
    m_g = (rg <= mu * xi_f) & (rg <= mc['r3'])
    ax.plot(rg[m_g], fg[m_g], 'k--', linewidth=lines_linewidth,
            label=rf'Gibbs tanh, $k_2={k2}$')
    ax.axhline(mc['f_maxwell'], color='gray', ls=':', lw=2.5,
               label=rf"Maxwell line $f^*={mc['f_maxwell']:.2f}$")

    r1, r2, r3 = mc['r1'], mc['r2'], mc['r3']
    fstar = mc['f_maxwell']
    m12 = (r >= r1) & (r <= r2)
    m23 = (r >= r2) & (r <= r3)
    ax.fill_between(r[m12], fstar, f[m12], color='red', alpha=0.25)
    ax.fill_between(r[m23], f[m23], fstar, color='blue', alpha=0.20)
    ax.annotate(r'$A_1$', ((r1+r2)/2, fstar + 0.35), fontsize=22, color='red')
    ax.annotate(r'$A_2$', ((r2+r3)/2, fstar - 0.55), fontsize=22, color='navy')

    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel(r'End-to-end distance $r$ / [$l_p$]', fontsize=label_fontsize)
    ax.set_ylabel(r'Force $f$ / [$k_BT/l_p$]', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, width=1,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=4, width=1,
                   direction=ytick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    ax.legend(fontsize=0.75*legend_fontsize, framealpha=0.9, edgecolor='none', loc='upper right')

    plt.tight_layout()
    fig.savefig(fpath, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('saved:', fpath)
    return mc


# ===================== 主程序 =====================
def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(outdir, exist_ok=True)

    print('=== N=1 单链有效力 ===')
    r1, u1 = single_domain_u()
    ch = Chain(u1, DR)
    print(f'支撑 [0, {ch.r[-1]:.3f}] lp, R0 = {ch.R0:.4f} lp')
    f_clip = np.clip(ch.f, 0.0, 1e4)

    # ---------- WLC 力拉伸边界 ----------
    x_w = np.linspace(1e-6, 1 - 1e-6, 2000)
    f_w = f_MS(x_w)
    r_folded   = x_w * xi_f          # 完全折叠轮廓长度
    r_unfolded = x_w * mu * xi_f     # 完全解折叠轮廓长度

    # ---------- Gibbs 唯象模型（截图参数：k1=6.5, k2=1.48） ----------
    # 力由 Marko-Siggia 关系参数化给出 f = f_MS(x)，
    # 轮廓长度 tanh 切换 Lc(f) = N*xi_f*{(mu+1)/2 + (mu-1)/2*tanh[k1*(f-k2)]}，
    # r(f) = x*Lc(f)。在 r/N 坐标下曲线与 N 无关，取 N=1 与单链图一致。
    k1_tanh, k2_tanh = 6.5, 1.48
    xg = np.linspace(1e-6, 1 - 1e-6, 4000)
    fg = f_MS(xg)
    Lc_g = xi_f * (0.5*(mu + 1) + 0.5*(mu - 1) * np.tanh(k1_tanh * (fg - k2_tanh)))
    rg = xg * Lc_g

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(ch.r, f_clip, 'r-', linewidth=lines_linewidth,
            label=r'$N=1$ protein chain')
    ax.plot(r_folded, f_w, 'b--', linewidth=lines_linewidth,
            label=r'WLC folded, $L=\xi_f$')
    ax.plot(r_unfolded, f_w, 'g-.', linewidth=lines_linewidth,
            label=r'WLC unfolded, $L=\mu\xi_f$')
    m_g = rg <= mu * xi_f
    ax.plot(rg[m_g], fg[m_g], 'k--', linewidth=lines_linewidth,
            label=r'Gibbs, $k_1=6.5$, $k_2=1.48$')

    ax.set_xlim(0.0, 28.0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel(r'End-to-end distance $r$ / [$l_p$]', fontsize=label_fontsize)
    ax.set_ylabel(r'Force $f$ / [$k_BT/l_p$]', fontsize=label_fontsize)
    ax.set_title('Single-chain force', fontsize=title_fontsize, pad=20)

    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, width=1,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=4, width=1,
                   direction=ytick_direction, right=ytick_right)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    ax.legend(fontsize=0.8*legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    plt.tight_layout()
    fpath = os.path.join(outdir, 'Single_chain_force_N1.png')
    fig.savefig(fpath, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('saved:', fpath)

    # ---------- Maxwell 等面积构造验证 + 示意图 ----------
    print('\n=== Maxwell 构造验证 ===')
    mc = maxwell_check(ch.r, f_clip, rg, fg, k2_tanh)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(ch.r, f_clip, 'r-', linewidth=lines_linewidth,
             label=r'$N=1$ protein chain')
    m_g2 = (rg <= mu * xi_f) & (rg <= mc['r3'])
    ax3.plot(rg[m_g2], fg[m_g2], 'k--', linewidth=lines_linewidth,
             label=r'Gibbs, $k_2=1.48$')
    ax3.axhline(mc['f_maxwell'], color='gray', ls=':', lw=2.5,
                label=rf"Maxwell line $f^*={mc['f_maxwell']:.2f}$")

    r1, r2, r3 = mc['r1'], mc['r2'], mc['r3']
    fstar = mc['f_maxwell']
    m12 = (ch.r >= r1) & (ch.r <= r2)
    m23 = (ch.r >= r2) & (ch.r <= r3)
    ax3.fill_between(ch.r[m12], fstar, f_clip[m12], color='red', alpha=0.25)
    ax3.fill_between(ch.r[m23], f_clip[m23], fstar, color='blue', alpha=0.20)
    ax3.annotate(r'$A_1$', ((r1+r2)/2, fstar + 0.35), fontsize=22, color='red')
    ax3.annotate(r'$A_2$', ((r2+r3)/2, fstar - 0.55), fontsize=22, color='navy')

    ax3.set_xlim(0.0, 14.0)
    ax3.set_ylim(0.0, 5.0)
    ax3.set_xlabel(r'End-to-end distance $r$ / [$l_p$]', fontsize=label_fontsize)
    ax3.set_ylabel(r'Force $f$ / [$k_BT/l_p$]', fontsize=label_fontsize)
    ax3.set_title('Maxwell construction check', fontsize=title_fontsize, pad=20)
    ax3.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax3.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax3.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    ax3.minorticks_on()
    ax3.tick_params(axis='x', which='minor', length=4, width=1,
                    direction=xtick_direction, top=xtick_top)
    ax3.tick_params(axis='y', which='minor', length=4, width=1,
                    direction=ytick_direction, right=ytick_right)
    for spine in ax3.spines.values():
        spine.set_linewidth(axes_linewidth)
    ax3.legend(fontsize=0.75*legend_fontsize, framealpha=0.9, edgecolor='none', loc='upper right')

    plt.tight_layout()
    fpath3 = os.path.join(outdir, 'Maxwell_check.png')
    fig3.savefig(fpath3, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('saved:', fpath3)

    # ---------- 图 4：多分散能垒系综的力-拉伸曲线 ----------
    print('\n=== 多分散能垒系综 ===')
    DEs = sample_delta_E()
    print(f'Delta_E 采样: min={DEs.min():.3f}, max={DEs.max():.3f}, '
          f'mean={DEs.mean():.3f}, std={DEs.std():.3f}')
    F_all = polydisperse_force(ch.r, DEs)
    f_avg = F_all.mean(axis=0)
    f_std = F_all.std(axis=0)
    i_pk_avg = int(np.argmax(np.where(ch.r < 12.0, f_avg, -np.inf)))
    print(f'平均力曲线: 峰 f={f_avg[i_pk_avg]:.3f} @ r={ch.r[i_pk_avg]:.3f}')

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for k in range(len(DEs)):
        ax4.plot(ch.r, F_all[k], '-', color='0.65', lw=0.7, alpha=0.45,
                 zorder=1)
    ax4.plot(ch.r, f_avg, 'r-', linewidth=lines_linewidth,
             label=r'ensemble average', zorder=3)
    ax4.plot(r_folded, f_w, 'b--', linewidth=lines_linewidth,
             label=r'WLC folded, $L=\xi_f$', zorder=2)
    ax4.plot(r_unfolded, f_w, 'g-.', linewidth=lines_linewidth,
             label=r'WLC unfolded, $L=\mu\xi_f$', zorder=2)
    m_g3 = rg <= mu * xi_f
    ax4.plot(rg[m_g3], fg[m_g3], 'k--', linewidth=lines_linewidth,
             label=r'Gibbs, $k_2=1.48$', zorder=2)

    ax4.set_xlim(0.0, 28.0)
    ax4.set_ylim(0.0, 7.0)
    ax4.set_xlabel(r'End-to-end distance $r$ / [$l_p$]', fontsize=label_fontsize)
    ax4.set_ylabel(r'Force $f$ / [$k_BT/l_p$]', fontsize=label_fontsize)
    ax4.set_title(r'Polydisperse barriers $\Delta E\sim\mathcal{N}(11.9,1.7^2)$',
                  fontsize=title_fontsize, pad=20)
    ax4.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax4.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax4.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    ax4.minorticks_on()
    ax4.tick_params(axis='x', which='minor', length=4, width=1,
                    direction=xtick_direction, top=xtick_top)
    ax4.tick_params(axis='y', which='minor', length=4, width=1,
                    direction=ytick_direction, right=ytick_right)
    for spine in ax4.spines.values():
        spine.set_linewidth(axes_linewidth)
    ax4.legend(fontsize=0.75*legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    plt.tight_layout()
    fpath4 = os.path.join(outdir, 'Single_chain_force_polydisperse.png')
    fig4.savefig(fpath4, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('saved:', fpath4)

    # ---------- 图 5：多分散平均曲线的 Maxwell 构造验证 ----------
    print('\n=== 多分散平均曲线的 Maxwell 构造验证 ===')
    plot_maxwell_figure(ch.r, f_avg, rg, fg, k2_tanh,
                        'Maxwell construction (polydisperse average)',
                        os.path.join(outdir, 'Maxwell_check_polydisperse.png'))


if __name__ == '__main__':
    main()
