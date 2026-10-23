"""
两种 delta-type 3-chain 网络的初始模量随 xi_f 的变化（kBT = lp = 1）。

对同一个 xi_f，考察两种 R0 规定：
  (A) R0 = R0_shared（固定常数，默认 2.0）；
  (B) R0 = R0_WLC(xi_f)（使用图片中的 WLC 解析均方末端距公式）：
      <r^2>_0 = 2 Lc lp * { 1 - (lp/Lc) * [1 - exp(-Lc/lp)] }
      R0 = sqrt(<r^2>_0) \approx sqrt(2 * N * xi_f * lp)

对每种 R0 规定，分别用两套公式：
  Case 1（展开网络，sec. 4.3）：
      使用真实的 PMF 自由能 F~(r) = -ln Z(r)，
      p(r) = δ(r - R0)，E0 = (R0/2)[R0 F~''(R0) + F~'(R0)]。
  Case 2（经典 WLC 3-chain）：
      F(r) = xi_f phi(r/xi_f)，
      E0 = (R0/2)[R0 fMSp(x0)/xi_f + fMS(x0)]，x0 = R0/xi_f。

统一 R0 后，同组内两条曲线的差异完全来自"展开物理" vs "固定轮廓长度"。

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

# ============ 字体与绘图风格设置（与前面代码保持一致）============
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
Delta_E   = 11.9     # 内部态能量尺度（kBT）
R0_shared = 2.0      # (A) 两种 case 共用的参考端距（lp）
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

def f_MS_prime(x):
    """微观力对 x 的解析导数。"""
    x = np.clip(x, 0, 1 - 1e-10)
    return 0.5 * (1 - x)**(-3) + 1.0

def U_of_n(n):
    """内部变量 n 的势能项。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

# ===================== WLC 解析 R0 =====================
def R0_WLC(xi_f, N=1, lp=1.0):
    """
    使用图片中的解析公式计算 WLC 模型的均方根首末端距：
        <r^2>_0 = 2 * Lc * lp * { 1 - (lp / Lc) * [1 - exp(-Lc / lp)] }
        R0 = sqrt(<r^2>_0)
    其中 Lc = N * xi_f 为轮廓长度。
    """
    Lc = N * xi_f
    # 直接使用精确公式，避免大 Lc 时相减导致精度流失，使用展开式
    if Lc < 1e-4:
        r2 = Lc**2  # 极小 Lc 时退化为刚性杆
    else:
        r2 = 2 * Lc * lp - 2 * lp**2 * (1 - np.exp(-Lc / lp))
    return np.sqrt(r2)

# ===================== Case 1：真实 PMF 自由能 F~(r) =====================
def compute_Ft_spline_and_R0(xi_f):
    """
    计算真实的 PMF 自由能 F~(r) = -ln Z(r) 的 PCHIP spline，
    以及平衡均方根端距 R0_eq。
    """
    Lc_max = mu * xi_f - DELTA
    r = np.arange(1e-9, Lc_max, DR)

    n = np.linspace(0, 1, NPTS)
    dn = n[1] - n[0]
    Lc = xi_f * (1 + n * (mu - 1))
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

    # 使用真实的 PMF 自由能 F~ = -ln Z 构建样条
    step = max(1, len(r) // 3000)
    Ft_spline = PchipInterpolator(r[::step], (-lnZ)[::step])
    return Ft_spline, R0_eq

def E0_from_Ft_spline(Ft_spline, R0):
    """
    Case 1 初始模量公式（使用 PMF 自由能 F~）：
        E0 = (R0/2) [ R0 F~''(R0) + F~'(R0) ]
    """
    return 0.5 * R0 * (R0 * Ft_spline.derivative(2)(R0)
                       + Ft_spline.derivative(1)(R0))

# ===================== Case 2：经典 WLC 完美网络（解析）=====================
def E0_WLC_delta(xi_f, R0_ref):
    """
    Case 2 初始模量解析公式：
        E0 = (R0/2) [ R0 fMSp(x0)/xi_f + fMS(x0) ]，x0 = R0_ref/xi_f
    """
    x0 = np.clip(R0_ref / xi_f, 1e-6, 1 - 1e-10)
    return 0.5 * R0_ref * (R0_ref * f_MS_prime(x0) / xi_f + f_MS(x0))

# ===================== 主程序 =====================
def main():
    outdir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures"
    os.makedirs(outdir, exist_ok=True)

    xi_f_vals = np.concatenate([np.linspace(3, 20, 18), np.linspace(22, 40, 10)])

    E1_shared, E2_shared = [], []   # (A) R0 = R0_shared
    E1_wlc,    E2_wlc    = [], []   # (B) R0 = R0_WLC(xi_f)
    R0_wlc_list, R0_eq_list = [], []

    print(f"{'xi_f':>6} {'R0_shared':>10} {'R0_WLC':>8} {'R0_approx':>10} {'R0_eq':>8}  "
          f"{'E1(sh)':>8} {'E2(sh)':>8} {'E1(wlc)':>8} {'E2(wlc)':>8}")
    for xi in xi_f_vals:
        Ft_spline, R0_eq = compute_Ft_spline_and_R0(xi)
        
        # 使用解析公式计算 R0_WLC
        R0_wlc = R0_WLC(xi)
        # 附带打印近似公式 sqrt(2 N xi_f lp)
        R0_approx = np.sqrt(2 * 1 * xi * 1.0)

        # (A) R0 = R0_shared
        e1_sh = E0_from_Ft_spline(Ft_spline, R0_shared)
        e2_sh = E0_WLC_delta(xi, R0_shared)

        # (B) R0 = R0_WLC
        e1_w = E0_from_Ft_spline(Ft_spline, R0_wlc)
        e2_w = E0_WLC_delta(xi, R0_wlc)

        E1_shared.append(e1_sh); E2_shared.append(e2_sh)
        E1_wlc.append(e1_w);     E2_wlc.append(e2_w)
        R0_wlc_list.append(R0_wlc); R0_eq_list.append(R0_eq)

        print(f"{xi:6.1f} {R0_shared:10.2f} {R0_wlc:8.3f} {R0_approx:10.3f} {R0_eq:8.3f}  "
              f"{e1_sh:8.3f} {e2_sh:8.3f} {e1_w:8.3f} {e2_w:8.3f}")

    E1_shared = np.array(E1_shared); E2_shared = np.array(E2_shared)
    E1_wlc    = np.array(E1_wlc);    E2_wlc    = np.array(E2_wlc)

    # ===================== 绘图 =====================
    fig, ax = plt.subplots(figsize=(11, 8))

    # (A) 固定 R0_shared：实心标记 + 实线
    ax.plot(xi_f_vals, E1_shared, 's-', color='blue',
            lw=lines_linewidth, ms=8,
            label=fr'Case 1 unfolding, $R_0={R0_shared:g}$')
    ax.plot(xi_f_vals, E2_shared, 'o-', color='green',
            lw=lines_linewidth, ms=8,
            label=fr'Case 2 WLC, $R_0={R0_shared:g}$')

    # (B) R0 = R0_WLC：空心标记 + 虚线
    ax.plot(xi_f_vals, E1_wlc, 's--', color='blue', mfc='white',
            lw=lines_linewidth, ms=8,
            label=r'Case 1 unfolding, $R_0 = R_0^{\rm WLC}$')
    ax.plot(xi_f_vals, E2_wlc, 'o--', color='green', mfc='white',
            lw=lines_linewidth, ms=8,
            label=r'Case 2 WLC, $R_0 = R_0^{\rm WLC}$')

    # 参考线
    ax.axhline(3.0, color='red', ls='--', lw=2, label='rubber limit 3')
    ax.axvline(3.6, color='gray', ls=':', lw=1.5)
    ax.text(3.8, 0.95, r'GB1: $\xi_f=3.6$',
            transform=ax.get_xaxis_transform(),
            fontsize=14, color='gray', va='top')

    ax.set_xlabel(r'Folded length $\xi_f\ [l_p]$')
    ax.set_ylabel(r'$E_0/(\rho k_BT)$')
    ax.set_title('Initial modulus of delta-type 3-chain networks ($N=1$)\n'
                 r'shared $R_0$ vs analytic WLC $R_0$')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, ls='--', alpha=grid_alpha)

    fig.tight_layout()
    fpath = os.path.join(outdir, 'G0_vs_xi_f_R0.png')
    fig.savefig(fpath, dpi=savefig_dpi, facecolor='white', edgecolor='none')
    print('\nsaved:', fpath)


if __name__ == '__main__':
    main()