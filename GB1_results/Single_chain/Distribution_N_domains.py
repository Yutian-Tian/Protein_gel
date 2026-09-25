import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid, simpson

# ================= 字体与绘图风格 =================
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 28
label_fontsize = 28
tick_fontsize = 24
legend_fontsize = 20

axes_linewidth = 2
grid_alpha = 0.4
lines_linewidth = 3
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
    'axes.titlesize': title_fontsize,
    'axes.labelsize': label_fontsize,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': legend_fontsize,
    'axes.linewidth': axes_linewidth,
    'grid.alpha': grid_alpha,
    'lines.linewidth': lines_linewidth,
    'figure.dpi': figure_dpi,
    'savefig.dpi': savefig_dpi,
    'xtick.direction': xtick_direction,
    'ytick.direction': ytick_direction,
    'xtick.top': xtick_top,
    'ytick.right': ytick_right,
})

# ================= 物理参数 =================
xi_f = 3.6
mu = 7.6
Delta_E = 11.7
k1 = 6.5
k2 = 1.50

# ================= 基础物理函数 =================
def phi(x):
    x = np.clip(x, 0, 0.999999)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def fc(x):
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def Lc_of_n(n):
    return xi_f * (1 + (mu - 1) * n)

def F_WLC(r, n):
    Lc = Lc_of_n(n)
    x = r / Lc
    x_clipped = np.clip(x, 0, 0.9999)
    res = (Lc / 4) * phi(x_clipped)
    return np.where(x >= 0.9999, 1e10, res)

def U_n(n):
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

def F_d(r, n):
    return F_WLC(r, n) + U_n(n)

# ================= 方法一：唯象力学积分 =================
def calc_phenomenological_single_domain():
    x_vals = np.linspace(0, 0.999, 2000)
    f_vals = fc(x_vals)
    n_vals = 0.5 * (1 + np.tanh(k1 * (f_vals - k2)))
    r_vals = x_vals * Lc_of_n(n_vals)
    F_eff = cumulative_trapezoid(f_vals, r_vals, initial=0)
    return r_vals, F_eff

def calc_phenomenological_chain(N, r_single, F_single):
    R_chain = N * r_single
    F_chain = N * F_single
    return R_chain, F_chain

# ================= 方法二：严格连续积分 =================
def calc_strict_single_domain_Feff(r_grid):
    """严格连续积分获得单域 F_eff(r)"""
    F_eff = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= xi_f * mu * 0.999:
            F_eff[i] = 1e10
            continue
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        res = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 1.0),
                              method='bounded')
        F_min = res.fun
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff[i] = 1e10
            continue
        n_vals = np.linspace(n_min, 1.0, 1000)
        integrand = np.exp(-(F_d(r, n_vals) - F_min))
        integral = simpson(integrand, x=n_vals)
        F_eff[i] = F_min - np.log(max(integral, 1e-300))
    return F_eff - F_eff[0]

# ============================================================
# 方法三：序贯蒙特卡洛（SMC / 粒子滤波）
# ============================================================
def calc_mc_chain_Feff_smc(N, M_samples, r_grid, F_eff_r,
                           bins=200, seed=42,
                           n_windows=50,
                           r_max_sample=None,
                           ESS_threshold=0.5,
                           verbose=True):
    """
    用序贯蒙特卡洛（SMC）计算 N 个 domain 串联链的自由能景观。

    核心思想
    --------
    逐 domain 采样 + 自适应重采样。在每一步 i 之后检查 ESS；
    若 ESS < ESS_threshold * M，则按权重对粒子重采样，并重置权重
    为均匀。这样避免了"一次性乘积 N 个权重"造成的权重退化。

    粒子状态
    --------
    R_vecs : (M, 3) 端到端矢量累积
    log_w  : (M,)  自上次重采样以来的累积对数权重

    每次迭代
    --------
    1. 采样 r_i（伞形窗口内均匀）和 û_i（球面均匀）
    2. R_vecs += r_i * û_i
    3. log_w += log P_eff(r_i) - log Q(r_i)
    4. 若 i < N-1：计算 ESS；低于阈值则重采样，重置 log_w

    参数
    ----
    ESS_threshold : float in (0, 1)
        自适应重采样阈值。ESS < ESS_threshold * M 时触发重采样。
        推荐 0.3 ~ 0.7。越小重采样越少（方差略大），越大越频繁（偏差略大）。
    """
    rng = np.random.default_rng(seed)

    if r_max_sample is None:
        r_max_sample = 0.95 * mu * xi_f

    # ---------- 目标分布对数密度 ----------
    F_shift = np.min(F_eff_r)
    P_eff_unnorm = 4 * np.pi * r_grid**2 * np.exp(-(F_eff_r - F_shift))
    Z_P = np.trapezoid(P_eff_unnorm, r_grid)
    log_P_eff = np.log(np.maximum(P_eff_unnorm / Z_P, 1e-300))

    # ---------- 窗口划分与 Q 的对数密度 ----------
    window_edges = np.linspace(0.0, r_max_sample, n_windows + 1)
    window_widths = window_edges[1:] - window_edges[:-1]
    # Q(r) = (1/K) / w_k 在窗口 k 内
    log_Q_per_window = np.log(1.0 / n_windows / window_widths)

    # ---------- 粒子状态初始化 ----------
    R_vecs = np.zeros((M_samples, 3))
    log_w = np.zeros(M_samples)          # 初始权重均匀

    ESS_target = M_samples * ESS_threshold
    n_resamples = 0
    ESS_last = float(M_samples)           # 用于日志

    # ============ 逐 domain 迭代 ============
    for i in range(N):
        # 1. 采样 r_i（伞形窗口内均匀）
        window_idx = rng.integers(0, n_windows, size=M_samples)
        r_lo = window_edges[window_idx]
        r_hi = window_edges[window_idx + 1]
        r_i = r_lo + rng.random(M_samples) * (r_hi - r_lo)

        # 2. 采样方向 û_i（球面均匀）
        v = rng.standard_normal((M_samples, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        # 3. 累积端到端矢量
        R_vecs += r_i[:, None] * v

        # 4. 累积对数权重
        log_P_i = np.interp(r_i, r_grid, log_P_eff,
                            left=log_P_eff[0], right=log_P_eff[-1])
        log_w += (log_P_i - log_Q_per_window[window_idx])

        # 5. 除最后一步外，检查 ESS 并在必要时重采样
        if i < N - 1:
            log_w_max = np.max(log_w)
            w = np.exp(log_w - log_w_max)
            w_sum = np.sum(w)
            w_norm = w / w_sum
            ESS = 1.0 / np.sum(w_norm ** 2)
            ESS_last = ESS

            if ESS < ESS_target:
                idx = rng.choice(M_samples, size=M_samples, p=w_norm)
                R_vecs = R_vecs[idx]
                log_w = np.zeros(M_samples)   # 重置权重为均匀
                n_resamples += 1

    # ---------- 最终权重 ----------
    log_w_max = np.max(log_w)
    w_chain = np.exp(log_w - log_w_max)

    # ---------- 端到端矢量模长 ----------
    R_mags = np.linalg.norm(R_vecs, axis=1)

    # ---------- 加权直方图 ----------
    hist_w, bin_edges = np.histogram(R_mags, bins=bins, weights=w_chain)
    counts, _ = np.histogram(R_mags, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    # ---------- 过滤与反演 ----------
    valid = (counts >= 2) & (hist_w > 1e-300)
    R_valid = bin_centers[valid]
    F_chain = -np.log(hist_w[valid] / shell_vol[valid])

    if R_valid.size < 5:
        if verbose:
            print(f"  [warn] N={N}: 有效 bin 只有 {R_valid.size} 个，回退 counts>=1")
        valid_fb = counts >= 1
        R_valid = bin_centers[valid_fb]
        F_chain = -np.log(hist_w[valid_fb] / shell_vol[valid_fb])

    if verbose:
        print(f"  [SMC] N={N}: 重采样 {n_resamples} 次, "
              f"最终 ESS ≈ {ESS_last:.0f} / {M_samples}")

    return R_valid, F_chain

# ================= 兼容旧接口 =================
def calc_mc_chain_Feff(N, M_samples, r_grid, F_eff_r,
                       bins=200, seed=42, min_counts=20,
                       n_windows=50, r_max_sample=None, ESS_min=None):
    """向后兼容：内部调用 SMC 版本。ESS_min 参数不再使用。"""
    return calc_mc_chain_Feff_smc(
        N=N, M_samples=M_samples, r_grid=r_grid, F_eff_r=F_eff_r,
        bins=bins, seed=seed, n_windows=n_windows,
        r_max_sample=r_max_sample, ESS_threshold=0.5)

# ================= 辅助：基线对齐 =================
def align_to_min(F):
    if F is None or F.size == 0:
        print("  [warn] align_to_min 收到空数组")
        return F
    return F - np.min(F)

# ================= 主图：单域 + N 域链对比 =================
def plot_N_comparison(output_dir):
    N_list = [1, 2, 4, 6, 8, 10]
    M_samples = 200000
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    seed = 42
    n_windows = 50
    ESS_threshold = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ---- 单域自由能 ----
    r_phe, F_phe = calc_phenomenological_single_domain()
    F_phe_aligned = align_to_min(F_phe)

    r_max_single = 0.98 * xi_f * mu
    r_grid = np.linspace(1e-4, r_max_single, 500)
    F_strict = calc_strict_single_domain_Feff(r_grid)
    F_strict_aligned = align_to_min(F_strict)

    ax1.plot(r_phe, F_phe_aligned, '--', color='darkorange',
             label='Phenomenological (Method 1)')
    ax1.plot(r_grid, F_strict_aligned, '-', color='purple',
             label='Strict (Method 3)')
    ax1.set_xlabel('End-to-end Length $r$', fontsize=label_fontsize)
    ax1.set_ylabel('$F_{\\text{eff}}(r)$', fontsize=label_fontsize)
    ax1.set_title('Single Domain Free Energy', fontsize=title_fontsize, pad=20)
    ax1.set_xlim(0, r_max_single)
    ax1.set_ylim(0, 30)
    ax1.grid(True, ls="--", alpha=grid_alpha)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none')

    # ---- 多域链自由能 ----
    for i, N in enumerate(N_list):
        R_phe_chain, F_phe_chain = calc_phenomenological_chain(N, r_phe, F_phe)
        F_phe_chain_aligned = align_to_min(F_phe_chain)

        R_mc_chain, F_mc_chain = calc_mc_chain_Feff_smc(
            N=N, M_samples=M_samples, r_grid=r_grid, F_eff_r=F_strict,
            bins=200, seed=seed, n_windows=n_windows,
            ESS_threshold=ESS_threshold, verbose=True)
        F_mc_chain_aligned = align_to_min(F_mc_chain)

        if F_mc_chain_aligned is None or F_mc_chain_aligned.size == 0:
            print(f"  [warn] N={N}: MC 结果为空，跳过绘制")
            continue

        ax2.plot(R_phe_chain, F_phe_chain_aligned, '--',
                 color=colors[i], label=f'Phenom. $N={N}$',
                 linewidth=2.5, alpha=0.8)
        ax2.plot(R_mc_chain, F_mc_chain_aligned, '-',
                 color=colors[i], label=f'MC SMC $N={N}$', linewidth=3)

    ax2.set_xlabel('Chain End-to-end Length $R$', fontsize=label_fontsize)
    ax2.set_ylabel('$\\Delta F_{\\text{chain}}(R)$', fontsize=label_fontsize)
    ax2.set_title('Free Energy Landscape vs $N$', fontsize=title_fontsize, pad=20)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 40)
    ax2.grid(True, ls="--", alpha=grid_alpha)
    ax2.legend(fontsize=16, framealpha=0.9, edgecolor='none')

    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', which='major', length=6,
                       direction=xtick_direction, top=xtick_top)
        ax.tick_params(axis='y', which='major',
                       direction=ytick_direction, right=ytick_right)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'F_chain_N_comparison.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white')
    print(f"图表已保存至: {save_path}")

# ================= 验证：N=1 应还原 F_eff(r) =================
def verify_N1(output_dir):
    r_max_single = 0.98 * xi_f * mu
    r_grid = np.linspace(1e-4, r_max_single, 500)
    F_strict = calc_strict_single_domain_Feff(r_grid)

    R_mc, F_mc = calc_mc_chain_Feff_smc(
        N=1, M_samples=500000, r_grid=r_grid, F_eff_r=F_strict,
        bins=250, seed=123, n_windows=30, ESS_threshold=0.5,
        verbose=True)

    if R_mc.size == 0:
        print("  [warn] verify_N1: MC 结果为空，跳过绘图")
        return

    F_strict_align = F_strict - np.min(F_strict)
    F_mc_align = F_mc - np.min(F_mc)

    F_strict_interp = np.interp(R_mc, r_grid, F_strict_align)
    offset = np.min(F_strict_interp) - np.min(F_mc_align)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_grid, F_strict_align, '-', color='purple', lw=3,
            label='Strict $F_{\\rm eff}(r)$')
    ax.plot(R_mc, F_mc_align + offset, '--', color='blue', lw=3,
            label='MC SMC $N=1$')
    ax.set_xlabel('End-to-end Length $r$', fontsize=label_fontsize)
    ax.set_ylabel('$F_{\\rm eff}(r)$', fontsize=label_fontsize)
    ax.set_title('Verification: $N=1$ vs $F_{\\rm eff}$',
                 fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, r_max_single)
    ax.set_ylim(0, 30)
    ax.grid(True, ls='--', alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize)

    save_path = os.path.join(output_dir, 'verify_N1.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white')
    print(f"验证图已保存至: {save_path}")

# ================= 主程序 =================
if __name__ == "__main__":
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Single_chain/results'
    plot_N_comparison(output_dir)
    verify_N1(output_dir)