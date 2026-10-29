"""
================================================================================
 蛋白质水凝胶网络初始剪切模量 G0 的计算与可视化
================================================================================
本脚本严格按 "Networks: N = 1" 图片给出的公式，计算并对比两种网络模型
的初始剪切模量 G0（单位：ρkBT）以及状态变量的边缘分布：

1. 可解折叠 (unfoldable) 蛋白质网络 (蓝色曲线)
   - 每条链存在折叠状态变量 n0 ∈ [0, 1]；
   - 轮廓长度严格依赖 n0：Lc(n0) = N·ξf·[1 + n0(μ−1)]；
   - 状态势能 U(n0) = ΔE·n0 − ΔE·cos(2πn0)；
   - 通过对 (n0, r0) 的双重积分，计入折叠/解折叠平衡的贡献。

2. 经典固定轮廓长度 WLC 网络 (绿色基线)
   - 单一状态 n0 = 0, U ≡ 0, Lc = N·ξf；
   - 作为不含可解折叠结构域时的对照参考。

核心公式（与图片一致）：
  (1) F_net(λ) = ∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0)
                     × [F̃(λr0) + 2 F̃(λ^{-1/2} r0)]
  (2) f(r)     = ∂F̃/∂r
  (3) σ(λ)    = ∂F_net/∂λ
                = ∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0) r0
                  × [f(λr0) − λ^{-3/2} f(λ^{-1/2} r0)]
  (4) G0      = dσ/dλ|_{λ=1}
                = (3/2) ∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0) r0
                  × [r0 f'(r0) + f(r0)]

参考态联合分布：
      p_r(r0,n0) ∝ exp{−[F̃(r0; Lc(n0)) + U(n0)]}，
      以 dr0·dn0 为测度归一化 (∫dn0 ∫dr0 p_r = 1)。

脚本输出：
   (1) G0 随折叠段长度 ξf 的变化曲线，并与橡胶弹性理论值 G0^Rubber = 3 对比；
   (2) 状态变量 n0 的边缘概率分布 p(n0)，用于分析折叠/解折叠平衡行为。

核心物理参数：μ = 7.6，ΔE = 11.9（单位：kBT）。
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ============ 字体与绘图风格设置 (保持不变) ============
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

trapz = getattr(np, 'trapezoid', np.trapz)   # numpy 版本兼容

# ============ 核心物理参数 ============
mu = 7.6
Delta_E = 11.9

# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 单链自由能无量纲部分: F̃(r;Lc) = Lc·φ(r/Lc), 且 φ'(x) = f_MS(x)"""
    x = np.clip(x, 0.0, 1.0 - 1e-12)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def f_MS(x):
    """图片式(2): f(r) = ∂F̃/∂r, 自变量 x = r/Lc (β=1, lp=1)"""
    x = np.clip(x, 0.0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def fcp(x):
    """df/dx (df/dr = fcp(x)/Lc) —— 对应图片的 f'(r)"""
    x = np.clip(x, 0.0, 0.999999)
    return 0.5 * (1 - x)**(-3) + 1.0

def Lc_of_n(n, N, xi_f):
    """轮廓长度 Lc(n0) = N·ξf·[1 + n0(μ-1)] (严格依赖状态变量 n0)"""
    return N * xi_f * (1 + n*(mu - 1))

def U_of_n(n):
    """状态势能 U(n0)"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

# ======================================================================
#  按 "Networks: N = 1" 图片逐条实现的网络模型
#  ────────────────────────────────────────────────────────────────────
#  (1) F_net(λ) = ∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0)[F̃(λr0)+2F̃(λ^{-1/2}r0)]
#  (2) f(r) = ∂F̃/∂r
#  (3) σ(λ) = ∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0) r0[f(λr0) − λ^{-3/2}f(λ^{-1/2}r0)]
#  (4) G0 = dσ/dλ|_{λ=1} = (3/2)∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0) r0[r0 f'(r0) + f(r0)]
#
#  p_r(r0,n0): 参考态(λ=1)联合分布, 取单链平衡 Boltzmann 权重
#      p_r ∝ exp{−[F̃(r0;Lc(n0)) + U(n0)]},
#      以 dr0·dn0 为测度归一化 (∫dn0∫dr0 p_r = 1), 与图片 "∫dr0 p_r" 的写法一致。
# ======================================================================
RADIAL_JACOBIAN = False   # True = 旧版代码的 r0² 权重 (三维径向分布测度)

def build_p_r(N, xi_f, unfoldable=True, n_points=401, r_points=6000):
    """
    构造并归一化联合分布 p_r(r0,n0) (图片三个公式共用):
        定义域 n0∈[0,1], r0∈[0, Lc(n0)], 测度 dr0·dn0, ∫dn0∫dr0 p_r = 1
    unfoldable=False → 经典 WLC 网络: 单一状态 (n0=0, U≡0, Lc=N·ξf)
    返回 dict: n_grid, r_grids, Lc, w(未归一化权重, p_r = w/Z), Z
    """
    if unfoldable:
        n_grid = np.linspace(0.0, 1.0, n_points)
    else:
        n_grid = np.array([0.0])

    r_grids, Lc_list, logw_list = [], [], []
    logw_max = -np.inf
    for n in n_grid:
        Lc = Lc_of_n(n, N, xi_f)
        r = np.linspace(0.0, Lc, r_points)       # 积分上限 Lc(n0), 与图片一致
        lw = -Lc * phi(r / Lc)                    # −F̃(r0)
        if unfoldable:
            lw = lw - U_of_n(n)                   # −U(n0)
        if RADIAL_JACOBIAN:
            lw = lw + 2.0 * np.log(np.clip(r, 1e-300, None))
        lw = np.nan_to_num(lw, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        r_grids.append(r); Lc_list.append(Lc); logw_list.append(lw)
        m = lw.max()
        if np.isfinite(m):
            logw_max = max(logw_max, m)

    w_list = [np.exp(lw - logw_max) for lw in logw_list]
    Z_n = np.array([trapz(w, r) for w, r in zip(w_list, r_grids)])
    Z = _dn_integral(Z_n, n_grid)
    return {'n_grid': n_grid, 'r_grids': r_grids, 'Lc': np.array(Lc_list),
            'w': w_list, 'Z': Z}

def _dn_integral(vals, n_grid):
    """沿 n0 的 trapezoid 积分 (单状态网格直接取值)"""
    if len(n_grid) == 1:
        return vals[0]
    return trapz(vals, n_grid)

def _expect(pre, g_list):
    """⟨g⟩ = ∫_0^1 dn0 ∫_0^{Lc(n0)} dr0 p_r(r0,n0) g(r0,n0)"""
    per_n = np.array([trapz(w * g, r)
                      for w, r, g in zip(pre['w'], pre['r_grids'], g_list)])
    return _dn_integral(per_n, pre['n_grid']) / pre['Z']

def F_net_of_lambda(pre, lam):
    """图片式(1): F_net(λ), 单位 kBT/链"""
    g_list = []
    for r, Lc in zip(pre['r_grids'], pre['Lc']):
        x_ax = lam * r / Lc            # 轴向
        x_tr = lam**-0.5 * r / Lc      # 两条横向
        g_list.append(Lc * phi(x_ax) + 2.0 * Lc * phi(x_tr))
    return _expect(pre, g_list)

def sigma_of_lambda(pre, lam):
    """图片式(3): σ(λ) = ∂F_net/∂λ"""
    g_list = []
    for r, Lc in zip(pre['r_grids'], pre['Lc']):
        g_list.append(r * (f_MS(lam * r / Lc)
                           - lam**-1.5 * f_MS(lam**-0.5 * r / Lc)))
    return _expect(pre, g_list)

def G0_of_network(pre):
    """
    与图片 σ(λ) 自洽的初始模量:
        G0 = dσ/dλ|_{λ=1} = (3/2)∫p_r r0[r0 f'(r0) + f(r0)]
    """
    g_list = []
    for r, Lc in zip(pre['r_grids'], pre['Lc']):
        x = r / Lc
        f0 = f_MS(x)          # f(r0)
        fp0 = fcp(x) / Lc     # f'(r0) = ∂f/∂r
        g = 1.5 * r * (r * fp0 + f0)
        g_list.append(g)
    return _expect(pre, g_list)

def G0_from_sigma_derivative(pre, h=1e-5):
    """自检: 数值 dσ/dλ|_{λ→1}, 应与 G0_of_network(pre) 一致"""
    return (sigma_of_lambda(pre, 1.0 + h) - sigma_of_lambda(pre, 1.0 - h)) / (2.0 * h)

# ===================== 两种网络模型的模量 =====================
def G0_3chain_unfoldable(N, xi_f, n_points=401, r_points=6000):
    """含可解折叠结构域的网络 (蓝色): 严格按图片公式, 双重积分 (n0, r0)"""
    pre = build_p_r(N, xi_f, unfoldable=True, n_points=n_points, r_points=r_points)
    return G0_of_network(pre)

def G0_3chain_WLC(N, xi_f, r_points=12000):
    """经典固定轮廓长 WLC 网络 (绿色基线): 同一套图片公式, 单一状态 (无 U(n))"""
    pre = build_p_r(N, xi_f, unfoldable=False, r_points=r_points)
    return G0_of_network(pre)

# ===================== 边缘分布 p(n0) =====================
def calculate_pn(N, xi_f, n_points=401, r_points=6000):
    """p(n0) = ∫ dr0 p_r(r0,n0)"""
    pre = build_p_r(N, xi_f, unfoldable=True, n_points=n_points, r_points=r_points)
    zeta = np.array([trapz(w, r) for w, r in zip(pre['w'], pre['r_grids'])])
    p_n = zeta / _dn_integral(zeta, pre['n_grid'])
    n_grid = pre['n_grid']
    if len(n_grid) > 1:
        print(f"  <n0> = {_dn_integral(n_grid * p_n, n_grid):.6g}")
    return n_grid, p_n

# ===================== 可视化函数 =====================
def plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=None):
    G0_quad_vals, G0_WLC_vals = [], []
    print(f"正在计算 N={N_single} 下的数据 ...")
    for xi_f in xi_f_vals:
        G0_quad_vals.append(G0_3chain_unfoldable(N_single, xi_f))
        G0_WLC_vals.append(G0_3chain_WLC(N_single, xi_f))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(0, max(xi_f_vals))
    ax.set_ylim(0, 15.0)

    ax.plot(xi_f_vals, G0_quad_vals, '--s', color='blue', markersize=7,
            label='Protein Networks', zorder=6)
    ax.plot(xi_f_vals, G0_WLC_vals, '--o', color='green', markersize=7,
            label='WLC Networks', zorder=5)
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=3,
               label='$G_0^{Rubber}=$3.0', zorder=1)

    ax.set_xlabel('Folded length $\\xi_\\mathrm{f}$', fontsize=label_fontsize)
    ax.set_ylabel('$G_0 / \\rho k_\\mathrm{B} T$', fontsize=label_fontsize)
    ax.set_title(f'Initial Modulus', fontsize=title_fontsize, pad=20)

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
    if save_dir:
        path = os.path.join(save_dir, f'Modulus_G0_vs_xi_f_N{N_single}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"图表已保存至: {path}")

def plot_pn_vs_n(N_single, xi_f, save_dir=None):
    """可视化边缘分布 p(n)"""
    n_grid, p_n = calculate_pn(N_single, xi_f)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.plot(n_grid, p_n, '-', color='purple', linewidth=4, label=f'$\\xi_\\mathrm{{f}}={xi_f}$')
    ax.plot(n_grid, p_n, 'o', color='purple', markersize=6, markerfacecolor='white')

    ax.set_xlabel('State variable $n$', fontsize=label_fontsize)
    ax.set_ylabel('Probability density $p(n)$', fontsize=label_fontsize)
    ax.set_title(f'Marginal Distribution ($N={N_single}$)', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, 1)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    # ---- major tick ----
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    # ---- minor tick ----
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=3, width=1,
                   direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='minor', length=3, width=1,
                   direction=ytick_direction, right=ytick_right)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    ax.legend(fontsize=legend_fontsize * 0.8, framealpha=0.9, edgecolor='none', loc='best')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'pn_distribution_N{N_single}_xi{xi_f}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"分布图已保存至: {path}")

# ===================== 主程序入口 =====================
def main():
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures'
    os.makedirs(output_dir, exist_ok=True)

    N_single = 1
    xi_f_vals = np.linspace(1.0, 100.0, 100)

    plot_G0_vs_xi_f(N_single, xi_f_vals, save_dir=output_dir)
    #plot_pn_vs_n(N_single, xi_f=3.6, save_dir=output_dir)
    #plot_pn_vs_n(N_single, xi_f=50.0, save_dir=output_dir)

if __name__ == "__main__":
    main()