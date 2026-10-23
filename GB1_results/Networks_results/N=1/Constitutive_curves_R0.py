"""
Sec. 4.3 (revised, PMF version) 本构曲线计算与可视化 (N=1, kBT = lp = 1)。

本脚本从完整代码中分离出单链 PMF 张力曲线与 delta 3-chain 网络本构曲线：
  1. 单链真实有效自由能（PMF）：F~(r) = -ln Z(r)
     Z(r) = \int dn exp(-Lc(n) phi(r/Lc(n)) - U(n))
     张力 f(r) = F~'(r)
  2. delta 3-chain 网络本构关系：
     sigma(lam) = (R0/3)[ f(lam R0) - lam^-3/2 f(lam^-1/2 R0) ]
     拉伸上限由完全展开不可伸长决定：lam_wall = (mu*xi_f - delta) / R0

参数固定为 xi_f = 3.6，对比 4 种 R0 预设：
     R0 = 0.25*xi_f, 0.5*xi_f, R0_eq(约0.577*xi_f), 0.75*xi_f
并叠加经典 WLC 3-chain (R0=0.5*xi_f) 作为对比基线。

约化单位：kBT = lp = 1。
"""
import os
import numpy as np
from scipy.special import logsumexp
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============ 字体与绘图风格设置（与代码2保持一致）============
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
legend_fontsize = 16

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 2.5
lines_markersize = 10

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
mu        = 7.6      # 展开态/折叠态轮廓长度比
Delta_E   = 11.7     # 内部态能量尺度（kBT）
xi_f      = 3.6      # 单个折叠域长度（lp）
DR        = 0.02     # 均匀网格间距（lp）
NPTS      = 800      # n 方向采样点数
DELTA     = 0.05     # 支撑区间保护：r < mu*xi_f - delta

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def f_MS(x):
    """微观力（Marko-Siggia），kBT = lp = 1。"""
    x = np.clip(x, 0, 1 - 1e-10)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def U_of_n(n):
    """内部变量 n 的势能项。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

# ===================== PMF 单链有效自由能计算 =====================
def compute_Ft_spline_and_R0(xi_f_val):
    """
    计算真实的 PMF 自由能 F~(r) = -ln Z(r) 的 PCHIP spline，
    以及平衡均方根端距 R0_eq。
    """
    Lc_max = mu * xi_f_val - DELTA
    r = np.arange(1e-9, Lc_max, DR)

    n = np.linspace(0, 1, NPTS)
    dn = n[1] - n[0]
    Lc = xi_f_val * (1 + n * (mu - 1))
    Un = U_of_n(n)

    # 条件配分函数：ln Z(r) = ln \int dn exp(-Lc(n) phi(r/Lc(n)) - U(n))
    lnZ = np.full(len(r), -np.inf)
    for j in range(NPTS):
        lnZ = np.logaddexp(lnZ, -Lc[j] * phi(r / Lc[j]) - Un[j])
    lnZ += np.log(dn)

    # 参考态端距分布 p0(r) ~ r^2 Z(r)
    ln_p0 = lnZ + 2 * np.log(r)
    ln_p0 -= logsumexp(ln_p0 + np.log(DR))
    p0 = np.exp(ln_p0)

    # 平衡均方根端距（诊断用）
    R0_eq = np.sqrt(trapz_func(p0 * r**2, r))

    # 构建真实 PMF 自由能样条
    step = max(1, len(r) // 3000)
    Ft_spline = PchipInterpolator(r[::step], (-lnZ)[::step])
    return Ft_spline, R0_eq

# ===================== 本构曲线计算 =====================
def calculate_constitutive(Fp, R0, Lc_max, num_pts=400):
    """
    计算给定 R0 下的 delta 3-chain 名义应力与拉伸比。
        sigma(lam) = (R0/3) * [ F'(lam*R0) - lam^-1.5 * F'(lam^-0.5*R0) ]
    上限截止于完全展开不可伸长（lam_wall = Lc_max / R0 的 98.5%）。
    """
    lam_wall = Lc_max / R0
    lam = np.linspace(1.0, 0.985 * lam_wall, num_pts)
    sig = (R0 / 3.0) * (Fp(lam * R0) - lam**-1.5 * Fp(lam**-0.5 * R0))
    return lam, sig, lam_wall

# ===================== 可视化函数 =====================
def plot_constitutive(Fp, R0_eq, xi_f_val, outdir):
    """
    绘制单链张力曲线与不同 R0 预设下的网络本构曲线。
    """
    Lc_max = mu * xi_f_val - DELTA
    r_plot = np.arange(0.05, Lc_max, 0.005)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- (a) 单链张力 F'(r) ---
    ax = axes[0]
    ax.plot(r_plot / xi_f_val, Fp(r_plot), 'b-', lw=lines_linewidth,
            label=r"$\tilde F'(r)$ (unfolding, PMF)")
    ax.plot(r_plot / xi_f_val, f_MS(r_plot / xi_f_val), 'g--', lw=2,
            label=r'WLC $f_{MS}(r/\xi_f)$')
    ax.axhline(0, color='gray', lw=1)
    ax.axhline(1.48, color='k', ls=':', lw=1.5)
    ax.text(0.15, 1.65, r'Gibbs $k_2=1.48$', fontsize=12)
    ax.axvline(mu, color='k', ls='--', lw=1.5)
    ax.text(mu - 0.4, 8.5, r'$\mu\xi_f$', fontsize=14)

    ax.set_xlabel(r'$r/\xi_f$')
    ax.set_ylabel(r'$k_BT/l_p$')
    ax.set_ylim(-0.3, 10)
    ax.set_xlim(0, mu + 0.3)
    ax.set_title(r'(a) Single-chain tension $\tilde F^\prime(r)$')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, ls='--', alpha=grid_alpha)

    # --- (b) 网络本构曲线 ---
    ax = axes[1]
    R0_list = [0.25 * xi_f_val, 0.5 * xi_f_val, R0_eq, 0.75 * xi_f_val]
    labels = [r'$R_0=0.25\xi_f$', r'$R_0=0.5\xi_f$',
              r'$R_0=R_0^{eq}\approx 0.577\xi_f$', r'$R_0=0.75\xi_f$']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

    print('\n本构曲线计算与诊断:')
    for R0, lab, c in zip(R0_list, labels, colors):
        lam, sig, lam_wall = calculate_constitutive(Fp, R0, Lc_max)

        ax.plot(lam, sig, color=c, lw=lines_linewidth, label=lab)
        ax.plot(lam[-1], sig[-1], 'o', color=c, ms=6)

        # 初始模量: E0 = (R0/2) * [R0 * F''(R0) + F'(R0)]
        F1 = Fp(R0)
        # 因为 Fp 是 spline 的一阶导，F'' 需通过 derivative 求
        # 但此处 Fp 已经是 PchipInterpolator.derivative(1)，可以直接再求导
        F2 = Fp.derivative(1)(R0) 
        e0 = 0.5 * R0 * (R0 * F2 + F1)

        print(f'R0 = {R0:6.3f} lp ({R0/xi_f_val:.3f} xi_f): '
              f'E0 = {e0:6.3f}, lam_wall = {lam_wall:6.2f}, sigma(end) = {sig[-1]:9.2f}')

    # 叠加 WLC 基线 (R0 = 0.5*xi_f)
    R0_wlc = 0.5 * xi_f_val
    lam_w = np.linspace(1.0, 0.97 * xi_f_val / R0_wlc, 200)
    sig_w = (R0_wlc / 3.0) * (
        f_MS(lam_w * R0_wlc / xi_f_val) - lam_w**-1.5 * f_MS(lam_w**-0.5 * R0_wlc / xi_f_val)
    )
    ax.plot(lam_w, sig_w, 'g--', lw=lines_linewidth,
            label=r'WLC 3-chain, $R_0=0.5\xi_f$ (wall at $\lambda=2$)')

    ax.set_xlabel(r'Stretch $\lambda$')
    ax.set_ylabel(r'$\sigma/(\rho k_BT)$')
    ax.set_xlim(1, 16)
    ax.set_ylim(0, 12)
    ax.set_title(r'(b) Sec.-4.3 revised constitutive curves ($N=1$)')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, ls='--', alpha=grid_alpha)

    fig.tight_layout()
    fpath = os.path.join(outdir, 'Constitutive_curves_R0.png')
    fig.savefig(fpath, dpi=savefig_dpi, facecolor='white', edgecolor='none')
    print('\nsaved:', fpath)


# ===================== 主程序 =====================
def main():
    outdir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures"
    os.makedirs(outdir, exist_ok=True)

    # 计算单链 PMF 及平衡 R0
    print(f'正在计算 xi_f = {xi_f} 的 PMF 自由能...')
    Ft_spline, R0_eq = compute_Ft_spline_and_R0(xi_f)
    Fp = Ft_spline.derivative(1)

    print(f'xi_f = {xi_f}, R0_eq = {R0_eq:.4f} lp = {R0_eq/xi_f:.3f} xi_f')

    # 调用绘图函数
    plot_constitutive(Fp, R0_eq, xi_f, outdir)


if __name__ == '__main__':
    main()