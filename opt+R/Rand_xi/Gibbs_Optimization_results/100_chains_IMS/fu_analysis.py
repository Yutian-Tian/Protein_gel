#!/usr/bin/env python3
"""
M条N个domain串联链的解折叠力(fu)统计分析程序
功能：
  1. 读取C++程序输出的CSV结果文件
  2. 识别每条链每个domain的解折叠力fu（f-r曲线的跃变点）
  3. 用高斯分布和广义极值分布(GEV)拟合fu的分布
  4. 科研风格双Y轴可视化：概率密度直方图+高斯拟合+GEV拟合+经验/理论CDF

作者: AI Assistant
日期: 2026-06-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path
import os
import warnings

# ============================================================
# 1. 科研风格可视化配置（Times New Roman + 高DPI）
# ============================================================

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    font_family = 'serif'

plt.rcParams.update({
    'font.family': font_family,
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.weight': 'normal',
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'legend.fontsize': 22,          # 稍调小以容纳更多图例
    'legend.title_fontsize': 22,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'grid.linewidth': 1,
    'grid.alpha': 0.4,
    'lines.linewidth': 5,
    'lines.markersize': 15,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# ============================================================
# 2. 数据读取函数
# ============================================================

def read_chain_csv(filepath):
    """读取C++输出的CSV文件"""
    df = pd.read_csv(filepath)
    f_vals = df.iloc[:, 0].values.astype(float)
    values = df.iloc[:, 1:].values.astype(float)
    return f_vals, values


def load_all_chain_results(data_dir, num_chains=100, num_domains=10):
    """加载所有链的CSV结果"""
    all_chains = []
    data_dir = Path(data_dir)

    for chain_idx in range(1, num_chains + 1):
        chain_data = {}
        r_file = data_dir / f"chain_{chain_idx}_r_values_unified.csv"
        n_file = data_dir / f"chain_{chain_idx}_n_values_unified.csv"
        Fd_file = data_dir / f"chain_{chain_idx}_Fd_values_unified.csv"
        x_file = data_dir / f"chain_{chain_idx}_x_values_unified.csv"

        for key, filepath in [('r', r_file), ('n', n_file),
                              ('Fd', Fd_file), ('x', x_file)]:
            if filepath.exists():
                f_vals, vals = read_chain_csv(filepath)
                chain_data['f'] = f_vals
                chain_data[key] = vals
            else:
                print(f"  [警告] 文件不存在: {filepath}")
                chain_data[key] = None

        all_chains.append(chain_data)
        if chain_idx % 20 == 0 or chain_idx == num_chains:
            print(f"  已加载 {chain_idx}/{num_chains} 条链...")

    return all_chains


# ============================================================
# 3. 解折叠力fu识别算法
# ============================================================

def detect_unfolding_force_jump(f_vals, r_vals, method='derivative',
                                threshold_factor=0.5, smoothing_window=5):
    """识别单个domain的fu"""
    n_points = len(f_vals)
    if n_points < 3:
        return None

    if not np.all(np.diff(f_vals) > 0):
        idx = np.argsort(f_vals)
        f_vals = f_vals[idx]
        r_vals = r_vals[idx]

    if method == 'derivative':
        dr_df = np.gradient(r_vals, f_vals)
        if smoothing_window > 1 and len(dr_df) > smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            dr_df_smooth = np.convolve(dr_df, kernel, mode='same')
        else:
            dr_df_smooth = dr_df

        max_jump_idx = np.argmax(dr_df_smooth)
        max_jump_val = dr_df_smooth[max_jump_idx]
        median_jump = np.median(np.abs(dr_df_smooth))

        if max_jump_val > threshold_factor * median_jump and max_jump_val > 0.01:
            return f_vals[max_jump_idx]

        d2r_df2 = np.gradient(dr_df_smooth, f_vals)
        zero_crossings = np.where(np.diff(np.sign(d2r_df2)) < 0)[0]
        if len(zero_crossings) > 0:
            best_idx = zero_crossings[np.argmax(dr_df_smooth[zero_crossings])]
            return f_vals[min(best_idx, len(f_vals)-1)]

    elif method == 'discontinuity':
        r_normalized = (r_vals - np.min(r_vals)) / (np.max(r_vals) - np.min(r_vals) + 1e-10)
        dr = np.diff(r_normalized)
        df = np.diff(f_vals)
        dr_df_norm = dr / (df + 1e-10)
        if len(dr_df_norm) > 0:
            jump_idx = np.argmax(dr_df_norm)
            return f_vals[jump_idx]

    return None


def detect_unfolding_force_multimethod(f_vals, r_vals, n_vals=None):
    """多方法联合识别fu"""
    fu_r = detect_unfolding_force_jump(f_vals, r_vals, method='derivative')

    fu_n = None
    if n_vals is not None and not np.all(np.isnan(n_vals)):
        dn_df = np.gradient(n_vals, f_vals)
        if len(dn_df) > 0:
            jump_idx = np.argmax(np.abs(dn_df))
            if np.abs(dn_df[jump_idx]) > 0.05:
                fu_n = f_vals[jump_idx]

    dr_df = np.gradient(r_vals, f_vals)
    d2r_df2 = np.gradient(dr_df, f_vals)
    if len(d2r_df2) > 0:
        curvature_idx = np.argmax(np.abs(d2r_df2))
        fu_curv = f_vals[curvature_idx]
    else:
        fu_curv = None

    candidates = [fu for fu in [fu_r, fu_n, fu_curv] if fu is not None]
    if len(candidates) == 0:
        return None
    return np.median(candidates)


def extract_all_unfolding_forces(all_chains, num_domains=10):
    """提取所有链的fu"""
    fu_list = []
    fu_by_chain = []

    for chain_idx, chain_data in enumerate(all_chains):
        if chain_data.get('r') is None or chain_data.get('f') is None:
            continue

        f_vals = chain_data['f']
        r_matrix = chain_data['r']
        n_matrix = chain_data.get('n')

        n_domains_actual = r_matrix.shape[1]
        chain_fu = []

        for domain_idx in range(n_domains_actual):
            r_vals = r_matrix[:, domain_idx]
            n_vals = n_matrix[:, domain_idx] if n_matrix is not None else None
            fu = detect_unfolding_force_multimethod(f_vals, r_vals, n_vals)
            if fu is not None:
                fu_list.append(fu)
                chain_fu.append(fu)

        fu_by_chain.append(chain_fu)
        if (chain_idx + 1) % 20 == 0 or chain_idx == len(all_chains) - 1:
            print(f"  已处理 {chain_idx + 1}/{len(all_chains)} 条链, "
                  f"累计 {len(fu_list)} 个fu值")

    return np.array(fu_list), fu_by_chain


# ============================================================
# 4. 分布拟合：高斯 + 广义极值分布 (GEV)
# ============================================================

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_to_fu(fu_data, n_bins=30):
    """高斯分布拟合（保持原接口，内部会调用直方图拟合）"""
    fu_data = np.array(fu_data)
    fu_data = fu_data[~np.isnan(fu_data)]
    if len(fu_data) == 0:
        raise ValueError("没有有效的fu数据")

    mu_init = np.mean(fu_data)
    sigma_init = np.std(fu_data)

    hist_counts, hist_edges = np.histogram(fu_data, bins=n_bins, density=True)
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0

    nonzero_mask = hist_counts > 0
    if np.sum(nonzero_mask) >= 2:
        popt, _ = curve_fit(gaussian_pdf, bin_centers[nonzero_mask],
                            hist_counts[nonzero_mask],
                            p0=[mu_init, sigma_init],
                            bounds=([mu_init - 3*sigma_init, 0.01*sigma_init],
                                    [mu_init + 3*sigma_init, 5*sigma_init]))
        mu_fit, sigma_fit = popt
    else:
        mu_fit, sigma_fit = mu_init, sigma_init

    return {
        'mu': mu_fit,
        'sigma': sigma_fit,
        'mu_std': mu_init,
        'sigma_std': sigma_init,
        'hist_counts': hist_counts,
        'hist_edges': hist_edges,
        'bin_centers': bin_centers,
    }


def fit_gev_to_fu(fu_data):
    """
    用MLE拟合广义极值分布（GEV）。
    返回参数：shape (ξ), loc (μ), scale (σ)
    注意：scipy的genextreme参数化：c = -ξ
    """
    fu_data = np.array(fu_data)
    fu_data = fu_data[~np.isnan(fu_data)]
    if len(fu_data) < 10:
        raise ValueError("数据点太少，无法可靠拟合GEV")

    # 用scipy直接MLE拟合
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # genextreme.fit 返回 (c, loc, scale) 其中 c = -ξ
        c, loc, scale = stats.genextreme.fit(fu_data)
    xi = -c          # 形状参数 ξ
    mu_gev = loc     # 位置参数
    sigma_gev = scale # 尺度参数

    return {
        'xi': xi,
        'mu': mu_gev,
        'sigma': sigma_gev,
        'c': c,
        'loc': loc,
        'scale': scale
    }


def get_gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    # 使用scipy的genextreme.pdf，注意参数转换
    return stats.genextreme.pdf(x, -xi, loc=mu, scale=sigma)


def get_gev_cdf(x, mu, sigma, xi):
    """GEV累积分布函数"""
    return stats.genextreme.cdf(x, -xi, loc=mu, scale=sigma)

# ===================== 独立封装的四个分布绘图函数 =====================
def plot_gaussian_pdf(ax, x, mu, sigma, 
                      color='blue', linewidth=4, linestyle='-', 
                      label=None, zorder=5):
    """
    在指定坐标轴上绘制高斯分布概率密度(PDF)曲线
    参数:
        ax: matplotlib坐标轴对象
        x: 横坐标数组
        mu: 高斯分布均值
        sigma: 高斯分布标准差
        其余为样式参数
    返回:
        line: 绘制的曲线对象
    """
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    line, = ax.plot(x, pdf, color=color, linewidth=linewidth, 
                    linestyle=linestyle, label=label, zorder=zorder)
    return line


def plot_gaussian_cdf(ax, x, mu, sigma, 
                      color='blue', linewidth=3, linestyle='--', 
                      label=None, zorder=4):
    """
    在指定坐标轴上绘制高斯分布累积分布(CDF)曲线
    """
    cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
    line, = ax.plot(x, cdf, color=color, linewidth=linewidth, 
                    linestyle=linestyle, label=label, zorder=zorder)
    return line


def plot_gev_pdf(ax, x, mu, sigma, xi, 
                 color='red', linewidth=4, linestyle='-', 
                 label=None, zorder=5):
    """
    在指定坐标轴上绘制GEV分布概率密度(PDF)曲线
    参数:
        xi: GEV形状参数(标准定义，scipy内部c=-xi)
    """
    pdf = stats.genextreme.pdf(x, -xi, loc=mu, scale=sigma)
    line, = ax.plot(x, pdf, color=color, linewidth=linewidth, 
                    linestyle=linestyle, label=label, zorder=zorder)
    return line


def plot_gev_cdf(ax, x, mu, sigma, xi, 
                 color='red', linewidth=3, linestyle='--', 
                 label=None, zorder=4):
    """
    在指定坐标轴上绘制GEV分布累积分布(CDF)曲线
    """
    cdf = stats.genextreme.cdf(x, -xi, loc=mu, scale=sigma)
    line, = ax.plot(x, cdf, color=color, linewidth=linewidth, 
                    linestyle=linestyle, label=label, zorder=zorder)
    return line

def prepare_fit_results(fu_data, n_bins=30):
    """
    准备所有拟合结果，生成统一的x网格用于绘图，
    并处理GEV支撑集限制。
    """
    # 高斯拟合
    gauss_res = fit_gaussian_to_fu(fu_data, n_bins)
    # GEV拟合
    gev_res = fit_gev_to_fu(fu_data)

    fu_data_clean = np.array(fu_data)
    fu_data_clean = fu_data_clean[~np.isnan(fu_data_clean)]

    # 构建合适的x拟合范围，考虑GEV的上界
    x_min = fu_data_clean.min() - 0.5 * gauss_res['sigma']
    x_max = fu_data_clean.max() + 0.5 * gauss_res['sigma']

    # 若GEV为Reverse Weibull型（ξ<0），上界为 mu - sigma/ξ
    if gev_res['xi'] < 0:
        upper_bound = gev_res['mu'] - gev_res['sigma'] / gev_res['xi']
        x_max = min(x_max, upper_bound * 1.001)  # 稍微留一点裕量

    x_fit = np.linspace(x_min, x_max, 500)

    # 计算高斯PDF/CDF
    pdf_gauss = gaussian_pdf(x_fit, gauss_res['mu'], gauss_res['sigma'])
    cdf_gauss = stats.norm.cdf(x_fit, loc=gauss_res['mu'], scale=gauss_res['sigma'])

    # 计算GEV PDF/CDF
    pdf_gev = get_gev_pdf(x_fit, gev_res['mu'], gev_res['sigma'], gev_res['xi'])
    cdf_gev = get_gev_cdf(x_fit, gev_res['mu'], gev_res['sigma'], gev_res['xi'])

    # 经验CDF
    fu_sorted = np.sort(fu_data_clean)
    empirical_cdf = np.arange(1, len(fu_sorted) + 1) / len(fu_sorted)

    # 理论CDF在数据点处的值（高斯和GEV）
    cdf_gauss_at_data = stats.norm.cdf(fu_sorted, loc=gauss_res['mu'], scale=gauss_res['sigma'])
    cdf_gev_at_data = get_gev_cdf(fu_sorted, gev_res['mu'], gev_res['sigma'], gev_res['xi'])

    return {
        'gauss': gauss_res,
        'gev': gev_res,
        'fu_data': fu_data_clean,
        'x_fit': x_fit,
        'pdf_gauss': pdf_gauss,
        'pdf_gev': pdf_gev,
        'cdf_gauss': cdf_gauss,
        'cdf_gev': cdf_gev,
        'fu_sorted': fu_sorted,
        'empirical_cdf': empirical_cdf,
        'cdf_gauss_at_data': cdf_gauss_at_data,
        'cdf_gev_at_data': cdf_gev_at_data,
    }


# ============================================================
# 5. 科研风格可视化（双Y轴，同时显示高斯和GEV）
# ============================================================

def plot_fu_distribution(fit_results, save_path=None, figsize=(10, 8), show_cdf=None):
    """
    主绘图函数：整合直方图、经验CDF、四个理论分布曲线
    参数:
        show_cdf: 是否开启右Y轴显示累积分布
    """
    fu_data = fit_results['fu_data']
    gauss = fit_results['gauss']
    gev = fit_results['gev']
    x_fit = fit_results['x_fit']

    fig, ax1 = plt.subplots(figsize=figsize)

    # ---- 左Y轴：概率密度 ----
    # 绘制直方图
    counts, edges, patches = ax1.hist(
        fu_data, bins=len(gauss['hist_counts']), density=True,
        color='lightcoral', edgecolor='black', alpha=0.4, linewidth=1.2,
        zorder=2, label='Histogram')

    # 调用封装函数绘制两个PDF
    plot_gaussian_pdf(
        ax1, x_fit, gauss['mu'], gauss['sigma'],
        color='blue', linewidth=4, linestyle='-',
        label=f"Gaussian ($\\mu$={gauss['mu']:.2f}, $\\sigma$={gauss['sigma']:.2f})"
    )
    plot_gev_pdf(
        ax1, x_fit, gev['mu'], gev['sigma'], gev['xi'],
        color='red', linewidth=4, linestyle='-',
        label=f"GEV ($\\mu$={gev['mu']:.2f}, $\\sigma$={gev['sigma']:.2f}, $\\xi$={gev['xi']:.2f})"
    )

    ax1.set_xlabel(r'Transition force $f_u$', fontsize=35)
    ax1.set_ylabel('Probability density', fontsize=35, color='black')
    ax1.tick_params(axis='y', labelcolor='black', direction='in',
                    top=True, right=not show_cdf, width=2, length=10)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True,
                    width=2, length=10)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='minor', direction='in',
                    top=True, right=not show_cdf, width=1.5, length=5)

    max_density = max(np.max(fit_results['pdf_gauss']),
                      np.max(fit_results['pdf_gev']),
                      np.max(counts) if len(counts) > 0 else 0)
    ax1.set_ylim(0, max_density * 1.2)

    # ---- 右Y轴：累积概率（可选开关） ----
    if show_cdf:
        ax2 = ax1.twinx()

        # 经验CDF
        ax2.plot(fit_results['fu_sorted'], fit_results['empirical_cdf'],
                 color='green', linewidth=4, linestyle='--',
                 label='Empirical CDF', zorder=4)

        # 调用封装函数绘制两个CDF
        plot_gaussian_cdf(
            ax2, x_fit, gauss['mu'], gauss['sigma'],
            color='blue', linewidth=3, linestyle='--',
            label='Gaussian CDF'
        )
        plot_gev_cdf(
            ax2, x_fit, gev['mu'], gev['sigma'], gev['xi'],
            color='red', linewidth=3, linestyle='--',
            label='GEV CDF'
        )

        ax2.set_ylabel('Cumulative probability', fontsize=35, color='black')
        ax2.tick_params(axis='y', labelcolor='black', direction='in',
                        top=True, right=True, width=2, length=10)
        ax2.set_ylim(0, 1.05)
        ax2.minorticks_on()
        ax2.tick_params(axis='y', which='minor', direction='in',
                        top=True, right=True, width=1.5, length=5)

        # 合并双轴图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper left', fontsize=20,
                   framealpha=0.9, edgecolor='gray',
                   fancybox=True, shadow=False)
        
        for spine in ax2.spines.values():
            spine.set_linewidth(2)
    else:
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1,
                   loc='upper left', fontsize=20,
                   framealpha=0.9, edgecolor='gray',
                   fancybox=True, shadow=False)

    # ---- 通用样式 ----
    ax1.set_title(r'$f_u$ Distribution: Gaussian vs. GEV', fontsize=38, pad=20)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1, zorder=0)

    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    x_min = fu_data.min() - 0.3 * gauss['sigma']
    x_max = x_fit.max()
    ax1.set_xlim(x_min, x_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  图形已保存至: {save_path}.png")

    return fig, (ax1, ax2) if show_cdf else (ax1,)


def plot_fu_by_chain(fu_by_chain, save_path=None, figsize=(16, 10)):
    """按链的散点图（保持不变）"""
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fu_by_chain)))
    for i, chain_fu in enumerate(fu_by_chain):
        if len(chain_fu) > 0:
            y_positions = [i] * len(chain_fu)
            ax.scatter(chain_fu, y_positions,
                       color=colors[i], s=80, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
    ax.set_xlabel(r'Transition force $f_u$', fontsize=35)
    ax.set_ylabel('Chain index', fontsize=35)
    ax.set_title(r'$f_u$ Distribution by Chain', fontsize=38, pad=20)
    ax.tick_params(direction='in', top=True, right=True, width=2, length=10)
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', top=True, right=True,
                   width=1.5, length=5)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_by_chain.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  链分布图已保存至: {save_path}_by_chain.png")

    return fig, ax


# ============================================================
# 6. 主程序
# ============================================================

def main(data_dir=None, num_chains=100, num_domains=10,
         n_bins=40, save_dir=None):
    if data_dir is None:
        data_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Gibbs_Optimization_results/100_chains_IMS/10_100_C_file"
    if save_dir is None:
        save_dir = data_dir

    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  M x N Domain链解折叠力(fu)统计分析（高斯 + GEV）")
    print("=" * 80)
    print(f"  数据目录: {data_dir}")
    print(f"  链数 M = {num_chains}")
    print(f"  Domain数 N = {num_domains}")
    print(f"  直方图bin数 = {n_bins}")
    print()

    # Step 1: 加载数据
    print("[Step 1/4] 加载所有链的CSV结果文件...")
    all_chains = load_all_chain_results(data_dir, num_chains, num_domains)
    print(f"  成功加载 {len(all_chains)} 条链\n")

    # Step 2: 提取fu
    print("[Step 2/4] 从f-r曲线中提取解折叠力fu...")
    fu_list, fu_by_chain = extract_all_unfolding_forces(all_chains, num_domains)
    print(f"  成功提取 {len(fu_list)} 个fu值")
    print(f"  fu范围: [{fu_list.min():.4f}, {fu_list.max():.4f}]")
    print(f"  fu均值: {fu_list.mean():.4f}")
    print(f"  fu标准差: {fu_list.std():.4f}\n")

    fu_save_path = save_dir / "unfolding_forces_raw.csv"
    np.savetxt(fu_save_path, fu_list, delimiter=',',
               header='unfolding_force_fu', comments='')
    print(f"  fu原始数据已保存至: {fu_save_path}\n")

    # Step 3: 高斯 + GEV 拟合
    print("[Step 3/4] 对fu分布进行高斯和GEV拟合...")
    fit_results = prepare_fit_results(fu_list, n_bins=n_bins)
    gauss = fit_results['gauss']
    gev = fit_results['gev']

    print("  高斯拟合结果:")
    print(f"    μ = {gauss['mu']:.6f}, σ = {gauss['sigma']:.6f}")
    print("  GEV拟合结果:")
    print(f"    μ = {gev['mu']:.6f}, σ = {gev['sigma']:.6f}, ξ = {gev['xi']:.6f}")
    if gev['xi'] < 0:
        upper = gev['mu'] - gev['sigma'] / gev['xi']
        print(f"    上界 = {upper:.6f} (Reverse Weibull 型)")
    print()

    # 保存拟合参数
    params_save_path = save_dir / "distribution_fit_parameters.csv"
    with open(params_save_path, 'w') as f:
        f.write("Distribution,Parameter,Value\n")
        f.write(f"Gaussian,mu,{gauss['mu']:.10f}\n")
        f.write(f"Gaussian,sigma,{gauss['sigma']:.10f}\n")
        f.write(f"GEV,mu,{gev['mu']:.10f}\n")
        f.write(f"GEV,sigma,{gev['sigma']:.10f}\n")
        f.write(f"GEV,xi,{gev['xi']:.10f}\n")
        f.write(f"Data,N_total,{len(fu_list)}\n")
        f.write(f"Data,N_chains,{num_chains}\n")
        f.write(f"Data,N_domains,{num_domains}\n")
    print(f"  拟合参数已保存至: {params_save_path}\n")

    # Step 4: 可视化
    print("[Step 4/4] 生成科研风格可视化...")
    viz_save_path = save_dir / "fu_distribution_gaussian_vs_gev"
    fig, axes = plot_fu_distribution(fit_results, save_path=str(viz_save_path))

    viz_chain_path = save_dir / "fu_distribution_by_chain"
    fig2, ax2 = plot_fu_by_chain(fu_by_chain, save_path=str(viz_chain_path))

    print("\n" + "=" * 80)
    print("  分析完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - {viz_save_path}.png                : fu分布（高斯 vs GEV）")
    print(f"  - {viz_chain_path}_by_chain.png      : 各链fu分布散点图")
    print(f"  - {fu_save_path.name}                : fu原始数据")
    print(f"  - {params_save_path.name}            : 高斯+GEV拟合参数")

    return fit_results, fu_list, fu_by_chain


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='M x N Domain链解折叠力(fu)统计分析')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()
    main(data_dir=args.data_dir, num_chains=args.M, num_domains=args.N,
         n_bins=args.bins, save_dir=args.save_dir)