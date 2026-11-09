"""N=1 退火有效自由能网络的初始模量 G0 与 xi_f。

F_eff(r)=-ln integral exp[-F_WLC(r,n)-U(n)]dn；kBT=lp=1。
f_eff=<f_n>，f_eff'=<f_n'>-Var(f_n)。
p0(r) ∝ r² exp[-F_eff(r)]，使用完整平衡径向分布，无需有限拉伸尾部截断。
G0/(rho*kBT)=1/2 <r² f_eff' + r f_eff>，rho 是链密度。
沿用 G0 名称，但这是初始单轴拉伸模量；不可压缩剪切模量为 G0/3。
固定内部态结果仅作涨落修正的诊断列，不作为退火模型。
第17页模型共享p0，用G0=1/2<r²f_G'+rf_G>计算，k1=6.5、k2=1.48。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
from scipy.special import logsumexp
from functools import lru_cache

# ============ 字体与绘图风格设置 (对齐代码2) ============
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
lines_markersize = 7

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

# ============ 核心物理参数 ============
mu = 7.6
Delta_E = 11.9

# ===================== 基础物理函数 =====================
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
    """
    修正：轮廓长度 Lc 必须严格依赖状态变量 n (图片第一张图)，而非力 f
    """
    return N * xi_f * (1 + n*(mu - 1))

def U_of_n(n):
    """图片中显式定义的势能项 U(n)"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

@lru_cache(maxsize=None)
def gauss_unit(points):
    q, w = np.polynomial.legendre.leggauss(points)
    return (q + 1) / 2, w / 2


def annealed_statistics(r, N, xi_f, n_points=512):
    """logZ、平均力、退火刚度、固定态平均刚度及条件力方差。"""
    r = np.asarray(r)
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


def modulus_components(pre):
    r, w, f = pre['r'], pre['mass'], pre['f']
    frozen = 0.5 * np.sum(w * (r*r*pre['frozen_fp'] + r*f))
    correction = 0.5 * np.sum(w * r*r*pre['variance'])
    return frozen - correction, frozen, correction


def G0_3chain_quad(N, xi_f, n_points=512, r_points=1024):
    """保持原函数接口；返回退火有效自由能的初始拉伸模量。"""
    return modulus_components(build_annealed(N, xi_f, n_points, r_points))[0]


def G0_3chain_WLC(N, xi_f, points=1024):
    L = N * xi_f
    x, wx = gauss_unit(points)
    logmass = 2*np.log(x) - L*phi(x) + np.log(wx)
    mass = np.exp(logmass - logsumexp(logmass))
    return 0.5 * L * np.sum(mass * x * (x*fcp(x) + f_MS(x)))


# 第17页模型：参考态与退火蛋白网络相同，变形响应由r(f)=x(f)Lc(f)指定。
K1 = 6.5
K2 = 1.48


def gibbs_force_stiffness(r, N, xi_f):
    """返回f_G及df_G/dr；推导见derivations/gibbs_initial_modulus.md。"""
    r = np.asarray(r, dtype=float)
    if np.any(r < 0) or np.any(r >= N*mu*xi_f):
        raise ValueError('gibbs distance must be below maximum contour length')
    lo, hi = np.zeros_like(r), np.ones_like(r)
    for _ in range(60):
        x = (lo + hi) / 2
        f = f_MS(x)
        t = np.tanh(K1*(f-K2))
        L = N*xi_f*((mu+1)/2 + (mu-1)*t/2)
        below = x*L < r
        lo, hi = np.where(below,x,lo), np.where(below,hi,x)
    x = (lo + hi) / 2
    f = f_MS(x)
    t = np.tanh(K1*(f-K2))
    L = N*xi_f*((mu+1)/2 + (mu-1)*t/2)
    Lp = N*xi_f*(mu-1)*K1*(1-t*t)/2
    fp = fcp(x)/(L + x*Lp*fcp(x))
    return f, fp


def G0_3chain_gibbs(pre):
    r, w = pre['r'], pre['mass']
    f, fp = gibbs_force_stiffness(r, pre['N'], pre['xi_f'])
    return 0.5*np.sum(w*(r*r*fp+r*f))


def validate_gibbs_modulus():
    lines = []
    for xi in [1.,3.6,10.,100.]:
        pre = build_annealed(1,xi)
        g = G0_3chain_gibbs(pre)
        fine = G0_3chain_gibbs(build_annealed(1,xi,n_points=1024,r_points=2048))
        assert np.isclose(g,fine,rtol=1e-6,atol=1e-7), (xi,g,fine)
        keep = pre['mass'] > 1e-22
        r,w = pre['r'][keep], pre['mass'][keep]
        def stress(lam):
            f1 = gibbs_force_stiffness(lam*r,1,xi)[0]
            f2 = gibbs_force_stiffness(lam**-0.5*r,1,xi)[0]
            return np.sum(w*r*(f1-lam**-1.5*f2))/3
        h=1e-5
        slope=(stress(1+h)-stress(1-h))/(2*h)
        assert stress(1.) == 0.
        assert np.isclose(g,slope,rtol=1e-6,atol=1e-7)
        # 单链刚度独立中心差分，覆盖过渡段及高伸长段。
        rr=np.linspace(.001,.95,200)*mu*xi
        f,fp=gibbs_force_stiffness(rr,1,xi)
        dh=1e-6*xi
        fd=(gibbs_force_stiffness(rr+dh,1,xi)[0]-
            gibbs_force_stiffness(rr-dh,1,xi)[0])/(2*dh)
        assert np.allclose(fp,fd,rtol=1e-6,atol=1e-7)
        line=f'gibbs xi_f={xi:g}: G0={g:.10g}, refined={fine:.10g}, sigma_derivative={slope:.10g}'
        print(line);lines.append(line)
    return lines


def calculate_pn(N, xi_f, n_points=512, r_points=1024):
    n, wn = gauss_unit(n_points)
    t, wt = gauss_unit(r_points)
    L = Lc_of_n(n, N, xi_f)
    # 对每个 n 完整积分 r² exp[-F_d] dr。
    lw = (3*np.log(L[:, None]) + 2*np.log(t)[None, :] -
          L[:, None]*phi(t)[None, :] - U_of_n(n)[:, None] + np.log(wt)[None, :])
    logzn = logsumexp(lw, axis=1)
    pn = np.exp(logzn - logsumexp(logzn + np.log(wn)))
    print(f'xi_f={xi_f:g}: <n>={np.sum(wn*n*pn):.8g}')
    return n, pn


def validate_modulus():
    lines = []
    for xi in [1., 3.6, 10., 100.]:
        pre = build_annealed(1, xi)
        g, frozen, correction = modulus_components(pre)
        finer = G0_3chain_quad(1, xi, n_points=1024, r_points=2048)
        assert np.isclose(g, finer, rtol=1e-6, atol=1e-7), (xi,g,finer)
        # 平衡径向分布的分部积分恒等式，独立核对刚度积分。
        rf = pre['r'] * pre['f']
        identity = 0.5 * (np.sum(pre['mass'] * rf**2) - 9)
        assert np.isclose(np.sum(pre['mass'] * rf), 3, rtol=1e-6)
        assert np.isclose(g, identity, rtol=1e-6, atol=1e-7)
        # 可忽略尾部仅用于有限差分，生产模量使用完整积分。
        keep = pre['mass'] > 1e-22
        r, w = pre['r'][keep], pre['mass'][keep]
        def stress(lam):
            f1 = annealed_statistics(lam*r, 1, xi)[1]
            f2 = annealed_statistics(lam**-0.5*r, 1, xi)[1]
            return np.sum(w*r*(f1-lam**-1.5*f2))/3
        h=1e-5
        slope=(stress(1+h)-stress(1-h))/(2*h)
        assert np.isclose(g,slope,rtol=1e-6,atol=1e-7)
        assert correction >= 0 and g <= frozen + 1e-12
        line=(f'xi_f={xi:g}: G0={g:.10g}, frozen={frozen:.10g}, '
              f'variance_correction={correction:.10g}, refined={finer:.10g}, '
              f'sigma_derivative={slope:.10g}, identity={identity:.10g}')
        print(line); lines.append(line)
    return lines

# ===================== 可视化函数 =====================
def plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=None):
    G0_quad_vals = []
    G0_WLC_vals = []
    G0_gibbs_vals = []
    frozen_vals, correction_vals = [], []

    print(f"正在计算 N={N_single} 下的数据 (退火有效自由能)...")
    for xi_f in xi_f_vals:
        pre = build_annealed(N_single, xi_f)
        g, frozen, correction = modulus_components(pre)
        G0_gibbs_vals.append(G0_3chain_gibbs(pre))
        G0_quad_vals.append(g)
        frozen_vals.append(frozen)
        correction_vals.append(correction)
        G0_WLC_vals.append(G0_3chain_WLC(N_single, xi_f))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(min(xi_f_vals), max(xi_f_vals))
    ax.set_ylim(0, 1.08 * max(max(G0_quad_vals), max(G0_WLC_vals),
                              max(G0_gibbs_vals), 3))

    # ---- marker 抽稀与统一尺寸 ----
    # 100 个数据点下每 ~8 点显示一个 marker，避免重叠。
    markevery = max(1, len(xi_f_vals) // 20)
    marker_kw = dict(markersize=lines_markersize, markeredgewidth=1.0, markevery=markevery)

    # 红线：退火有效自由能的模量，含条件力方差修正
    ax.plot(xi_f_vals, G0_quad_vals, '-s', color='red',
            linewidth=lines_linewidth,
            label=f'$N={N_single}$ protein network',
            zorder=6, **marker_kw)

    # 蓝线：固定轮廓长度 WLC
    ax.plot(xi_f_vals, G0_WLC_vals, '-o', color='blue',
            linewidth=lines_linewidth,
            label='WLC network',
            zorder=5, **marker_kw)

    # 黑线：Gibbs network model
    ax.plot(xi_f_vals, G0_gibbs_vals, '-^', color='black',
            linewidth=lines_linewidth,
            label='Gibbs network model',
            zorder=7, **marker_kw)

    ax.axhline(y=3.0, color='gray', linestyle='--',
               linewidth=lines_linewidth,
               label='$G_0^{Rubber}=$3.0', zorder=1)

    ax.set_xlabel('Folded length $\\xi_\\mathrm{f}$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / \\rho k_\\mathrm{B} T$', fontsize=label_fontsize)
    ax.set_title(f'Initial Modulus', fontsize=title_fontsize, pad=20)

    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    # ---- major tick 设置 ----
    ax.tick_params(axis='x', which='major', length=6,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width,
                   direction=ytick_direction, right=ytick_right)
    # ---- minor tick 设置 ----
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

    if save_dir:
        path = os.path.join(save_dir, f'G0_vs_xi_f_N{N_single}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        np.savetxt(os.path.join(save_dir, f'G0_vs_xi_f_N{N_single}.csv'),
                   np.column_stack((xi_f_vals, G0_quad_vals, G0_WLC_vals,
                                    frozen_vals, correction_vals,
                                    G0_gibbs_vals)), delimiter=',',
                   header='xi_f_over_lp,G0_annealed_over_rho_kBT,'
                          'G0_WLC_over_rho_kBT,G0_frozen_over_rho_kBT,'
                          'variance_correction_over_rho_kBT,'
                          'G0_gibbs_over_rho_kBT',
                   comments='')
        print(f"图表已保存至: {path}")
    plt.close(fig)

# ===================== 可视化边缘分布p(n) =====================
def plot_pn_vs_n(N_single, xi_f, save_dir=None):
    """
    可视化边缘分布 p(n)
    """
    n_grid, p_n = calculate_pn(N_single, xi_f)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('log')

    # 绘制 p(n)
    ax.plot(n_grid, p_n, '-', color='purple', linewidth=lines_linewidth,
            label=f'$\\xi_\\mathrm{{f}}={xi_f}$')
    # 可以用散点强调
    ax.plot(n_grid, p_n, 'o', color='purple', markersize=6, markerfacecolor='white')

    ax.set_xlabel('State variable $n$', fontsize=label_fontsize)
    ax.set_ylabel('Probability density $p(n)$', fontsize=label_fontsize)
    ax.set_title(f'Marginal Distribution ($N={N_single}$)', fontsize=title_fontsize, pad=20)

    ax.set_xlim(0, 1)

    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    # ---- major tick 设置 ----
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    # ---- minor tick 设置 ----
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, width=1, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=4, width=1, direction=ytick_direction, right=ytick_right)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    ax.legend(fontsize=0.7*legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'pn_distribution_N{N_single}_xi{xi_f}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"分布图已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    N_single = 1
    xi_f_vals = np.linspace(1.0, 100.0, 100)
    validation = validate_modulus() + validate_gibbs_modulus()
    with open(os.path.join(output_dir, 'Modulus_N1_validation.txt'), 'w') as report:
        report.write(f'N=1, mu={mu}, Delta_E={Delta_E}, k1={K1}, k2={K2}, kBT=lp=1; full equilibrium reference\n')
        report.write('\n'.join(validation) + '\n')

    plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=output_dir)


if __name__ == "__main__":
    main()