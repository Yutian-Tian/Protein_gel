"""
单链力-伸长曲线对比（N=1, kBT = lp = 1）。

在同一张图中对比两种单链有效力的定义：
  (1) 条件平均自由能（原方案）：
        Fbar(r) = <F(r,n)>_{n|r}
        Fbar'(r) = d/dr <F(r,n)>_{n|r}
  (2) 真实 PMF（前文提及）：
        F~(r) = -ln Z(r) = -ln \int dn exp(-Lc(n) phi(r/Lc(n)) - U(n))
        F~'(r) = <f_MS(r/Lc(n))>_{n|r}   (条件平均微观力)
  附加：WLC 基线 f_MS(r/xi_f)，用于定位"固定轮廓长度"参考。

两条曲线都描述了同一个退火展开模型的自由能导数；差别来自
"先对 r 求导，后对 n 平均" 与 "先对 n 平均，后对 r 求导" 的顺序。

横轴为端距 r（单位 lp），不再做 r/xi_f 归一化。

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

# ============ 字体与绘图风格设置 ============
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

axes_linewidth = 2.5
xtick_major_width = 2.5
ytick_major_width = 2.5
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 3.5
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
xi_f      = 3.6      # 单个折叠域长度（lp），即 GB1 的折叠长度
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

# ===================== 单链自由能与力计算 =====================
def compute_single_chain_forces(xi_f_val):
    """
    计算单链两种有效力：
      (1) Fbar'(r) = d/dr <F(r,n)>_{n|r}         （条件平均自由能的导数）
      (2) F~'(r)   = -d/dr ln Z(r) = <f_MS>_{n|r}（PMF 的导数）
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

    # (1) 条件平均自由能 Fbar(r) = \sum_j p(n_j|r) F(r, n_j) * dn
    mF = np.zeros(len(r))
    for j in range(NPTS):
        F = Lc[j] * phi(r / Lc[j]) + Un[j]
        mF += np.exp(-F - lnZ) * dn * F

    # 平衡均方根端距（诊断用）
    R0_eq = np.sqrt(trapz_func(p0 * r**2, r))

    # 用 PCHIP 平滑，降采样以稳定一阶导
    step = max(1, len(r) // 3000)
    r_ds = r[::step]

    # 两种自由能的样条
    Fbar_spline = PchipInterpolator(r_ds, mF[::step])       # <F>_{n|r}
    Ft_spline   = PchipInterpolator(r_ds, (-lnZ)[::step])   # -ln Z

    # 对各自求一阶导 = 力
    Fbar_prime = Fbar_spline.derivative(1)
    Ft_prime   = Ft_spline.derivative(1)

    return Fbar_prime, Ft_prime, R0_eq, Lc_max

# ===================== 可视化函数 =====================
def plot_single_chain_forces(Fbar_prime, Ft_prime, Lc_max, xi_f_val, outdir):
    """
    绘制两种单链有效力-伸长曲线的对比图。
    横轴为端距 r（单位 lp）。
    """
    r_plot = np.arange(0.05, Lc_max, 0.005)

    fig, ax = plt.subplots(figsize=(8, 6))

    # (1) 条件平均自由能方案：Fbar'(r)
    ax.plot(r_plot, Fbar_prime(r_plot), 'b-', lw=lines_linewidth,
            label=r'$\langle F \rangle$')

    # (2) PMF 方案：F~'(r)
    ax.plot(r_plot, Ft_prime(r_plot), 'r-', lw=lines_linewidth,
            label=r'$\tilde{F}$')

    # WLC 基线（固定轮廓长度），自变量需为 r/xi_f
    ax.plot(r_plot, f_MS(r_plot / xi_f_val), 'g--', lw=lines_linewidth,
            label=f'$\\xi_f = {xi_f_val}$')
    ax.plot(r_plot, f_MS(r_plot / (mu*xi_f_val)), 'g-.', lw=lines_linewidth,
            label=f'$\\xi_f = {mu*xi_f_val}$')

    ax.set_xlabel(r'End-to-end distance $r\ [l_p]$', fontsize=label_fontsize)
    ax.set_ylabel(r'Force $f \quad [k_B T / l_p]$', fontsize=label_fontsize)
    ax.set_xlim(0, mu * xi_f_val * 1.02)
    ax.set_ylim(-1.5, 10)
    ax.set_title('Force-extension Relation', fontsize=title_fontsize, pad=20)
    
    # 图例的字体大小使用全局设置
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', fontsize=legend_fontsize)
    ax.grid(True, ls='--', alpha=grid_alpha)

    fig.tight_layout()
    fpath = os.path.join(outdir, 'single_chain_force_Fbar_vs_PMF.png')
    fig.savefig(fpath, dpi=savefig_dpi, facecolor='white', edgecolor='none')
    print('\nsaved:', fpath)


# ===================== 主程序 =====================
def main():
    outdir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures"
    os.makedirs(outdir, exist_ok=True)

    print(f'正在计算 xi_f = {xi_f} 的单链自由能与力...')
    Fbar_prime, Ft_prime, R0_eq, Lc_max = compute_single_chain_forces(xi_f)

    print(f'xi_f = {xi_f}, R0_eq = {R0_eq:.4f} lp = {R0_eq/xi_f:.3f} xi_f, '
          f'轮廓壁位于 {Lc_max:.2f} lp')

    # 诊断：在若干 r 处比较两种力
    print('\n   r [lp]   Fbar_prime   Ft_prime')
    for frac in [0.25, 0.5, 0.577, 0.75, 1.5, 3.0]:
        r_test = frac * xi_f
        print(f'{r_test:8.3f}   {Fbar_prime(r_test):10.4f}  {Ft_prime(r_test):10.4f}')

    plot_single_chain_forces(Fbar_prime, Ft_prime, Lc_max, xi_f, outdir)


if __name__ == '__main__':
    main()