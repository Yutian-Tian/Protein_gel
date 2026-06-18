"""
M条N个domain串联链的解折叠力(fu)统计分析程序
功能：
  1. 读取C++程序输出的CSV结果文件
  2. 识别每条链每个domain的解折叠力fu（f-r曲线的跃变点）
  3. 用高斯分布拟合fu的分布
  4. 科研风格双Y轴可视化：概率密度直方图+高斯拟合+经验/理论CDF

作者: AI Assistant
日期: 2026-06-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import interp1d
import os
import glob
from pathlib import Path

# ============================================================
# 1. 科研风格可视化配置（Times New Roman + 高DPI）
# ============================================================

# 尝试加载 Times New Roman
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    # 回退到系统默认的衬线字体
    font_family = 'serif'

plt.rcParams.update({
    'font.family': font_family,
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.weight': 'normal',
    # 字体大小
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'legend.fontsize': 25,
    'legend.title_fontsize': 25,
    # 线宽
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
    """
    读取C++程序输出的CSV文件。
    格式：第一行为列标题，第一列为力值f，后面N列为Domain_1到Domain_N的值。
    
    返回:
        f_vals: 力值数组 (n_points,)
        values: 各domain的值矩阵 (n_points, N_domains)
    """
    df = pd.read_csv(filepath)
    # 第一列是力值（没有列标题或标题为空）
    f_vals = df.iloc[:, 0].values.astype(float)
    # 后面的列为domain值
    values = df.iloc[:, 1:].values.astype(float)
    return f_vals, values


def load_all_chain_results(data_dir, num_chains=100, num_domains=10):
    """
    加载所有链的CSV结果文件。
    
    参数:
        data_dir: C++程序输出目录
        num_chains: 链的总数M
        num_domains: 每链domain数N
    
    返回:
        list of dict, 每个元素包含一条链的 {f_vals, r_values, n_values, Fd_values, x_values}
    """
    all_chains = []
    data_dir = Path(data_dir)
    
    for chain_idx in range(1, num_chains + 1):
        chain_data = {}
        
        # 读取四种数据文件
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
    """
    从单个domain的f-r曲线中识别解折叠力fu（跃变点）。
    
    策略：
      1. 计算dr/df（r对f的导数），找到峰值
      2. 或者检测n(f)从~0跳到~1的位置
      3. 或者检测x(f)的跃变
    
    参数:
        f_vals: 力值数组 (n_points,), 必须单调递增
        r_vals: 端到端距离数组 (n_points,)
        method: 'derivative' 或 'discontinuity'
        threshold_factor: 峰值阈值因子
        smoothing_window: 平滑窗口大小
    
    返回:
        fu: 解折叠力（标量），未找到则返回None
    """
    n_points = len(f_vals)
    if n_points < 3:
        return None
    
    # 确保f单调递增
    if not np.all(np.diff(f_vals) > 0):
        # 去重并排序
        idx = np.argsort(f_vals)
        f_vals = f_vals[idx]
        r_vals = r_vals[idx]
    
    # 方法1: 基于dr/df的峰值检测
    if method == 'derivative':
        # 计算dr/df
        dr_df = np.gradient(r_vals, f_vals)
        
        # 平滑处理
        if smoothing_window > 1 and len(dr_df) > smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            dr_df_smooth = np.convolve(dr_df, kernel, mode='same')
        else:
            dr_df_smooth = dr_df
        
        # 找到最大峰值
        max_jump_idx = np.argmax(dr_df_smooth)
        max_jump_val = dr_df_smooth[max_jump_idx]
        
        # 阈值筛选：峰值必须显著大于背景
        median_jump = np.median(np.abs(dr_df_smooth))
        if max_jump_val > threshold_factor * median_jump and max_jump_val > 0.01:
            fu = f_vals[max_jump_idx]
            return fu
        
        # 如果没有明显峰值，尝试找dr/df的拐点
        d2r_df2 = np.gradient(dr_df_smooth, f_vals)
        # 找d2r/df2从正变负的零点（dr/df的极大值点）
        zero_crossings = np.where(np.diff(np.sign(d2r_df2)) < 0)[0]
        if len(zero_crossings) > 0:
            # 选择dr/df最大的那个
            best_idx = zero_crossings[np.argmax(dr_df_smooth[zero_crossings])]
            fu = f_vals[min(best_idx, len(f_vals)-1)]
            return fu
    
    # 方法2: 基于n(f)跃变的检测
    elif method == 'discontinuity':
        # 计算r的归一化变化率
        r_normalized = (r_vals - np.min(r_vals)) / (np.max(r_vals) - np.min(r_vals) + 1e-10)
        
        # 找到最陡峭的上升段
        dr = np.diff(r_normalized)
        df = np.diff(f_vals)
        dr_df_norm = dr / (df + 1e-10)
        
        if len(dr_df_norm) > 0:
            jump_idx = np.argmax(dr_df_norm)
            fu = f_vals[jump_idx]
            return fu
    
    return None


def detect_unfolding_force_multimethod(f_vals, r_vals, n_vals=None):
    """
    多方法联合检测解折叠力，取最可靠的结果。
    
    参数:
        f_vals: 力值数组
        r_vals: 端到端距离数组
        n_vals: 展开分数数组（可选）
    
    返回:
        fu: 解折叠力
    """
    # 方法1: r(f)导数峰值
    fu_r = detect_unfolding_force_jump(f_vals, r_vals, method='derivative')
    
    # 方法2: 如果有n_vals，用n(f)跃变
    fu_n = None
    if n_vals is not None and not np.all(np.isnan(n_vals)):
        # n从~0跳到~1的位置
        dn_df = np.gradient(n_vals, f_vals)
        if len(dn_df) > 0:
            jump_idx = np.argmax(np.abs(dn_df))
            if np.abs(dn_df[jump_idx]) > 0.05:  # 显著跃变
                fu_n = f_vals[jump_idx]
    
    # 方法3: 二阶导数检测（曲率最大点）
    dr_df = np.gradient(r_vals, f_vals)
    d2r_df2 = np.gradient(dr_df, f_vals)
    if len(d2r_df2) > 0:
        # 找曲率绝对值最大的位置
        curvature_idx = np.argmax(np.abs(d2r_df2))
        fu_curv = f_vals[curvature_idx]
    else:
        fu_curv = None
    
    # 投票机制：选择两个方法都接近的值
    candidates = [fu for fu in [fu_r, fu_n, fu_curv] if fu is not None]
    
    if len(candidates) == 0:
        return None
    
    # 如果有多个候选，取中位数（更稳健）
    return np.median(candidates)


def extract_all_unfolding_forces(all_chains, num_domains=10):
    """
    从所有链的所有domain中提取解折叠力fu。
    
    参数:
        all_chains: load_all_chain_results()的输出
        num_domains: 每链domain数
    
    返回:
        fu_list: 所有解折叠力的列表 (M*N,)
        fu_by_chain: 按链分组的解折叠力 (M, N)
    """
    fu_list = []
    fu_by_chain = []
    
    for chain_idx, chain_data in enumerate(all_chains):
        if chain_data.get('r') is None or chain_data.get('f') is None:
            continue
        
        f_vals = chain_data['f']
        r_matrix = chain_data['r']  # (n_points, N_domains)
        n_matrix = chain_data.get('n')  # (n_points, N_domains)
        
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
# 4. 高斯分布拟合
# ============================================================

def gaussian_pdf(x, mu, sigma):
    """高斯概率密度函数。"""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_cdf(x, mu, sigma):
    """高斯累积分布函数。"""
    return 0.5 * (1 + stats.norm.cdf((x - mu) / sigma) * 2 - 1)
    # 等价于 scipy.stats.norm.cdf(x, loc=mu, scale=sigma)


def fit_gaussian_to_fu(fu_data, n_bins=30):
    """
    对fu数据拟合高斯分布。
    
    参数:
        fu_data: 解折叠力数组
        n_bins: 直方图bin数
    
    返回:
        dict: 包含mu, sigma, hist_counts, hist_edges, bin_centers, x_fit等
    """
    fu_data = np.array(fu_data)
    fu_data = fu_data[~np.isnan(fu_data)]  # 去除NaN
    
    if len(fu_data) == 0:
        raise ValueError("没有有效的fu数据用于拟合")
    
    # 经验均值和标准差作为初始猜测
    mu_init = np.mean(fu_data)
    sigma_init = np.std(fu_data)
    
    # 直方图
    hist_counts, hist_edges = np.histogram(fu_data, bins=n_bins, density=True)
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
    
    # 用直方图数据拟合高斯PDF
    # 只使用非零bin进行拟合
    nonzero_mask = hist_counts > 0
    if np.sum(nonzero_mask) >= 2:
        popt, pcov = curve_fit(gaussian_pdf, 
                               bin_centers[nonzero_mask], 
                               hist_counts[nonzero_mask],
                               p0=[mu_init, sigma_init],
                               bounds=([mu_init - 3*sigma_init, 0.01*sigma_init],
                                       [mu_init + 3*sigma_init, 5*sigma_init]))
        mu_fit, sigma_fit = popt
    else:
        mu_fit, sigma_fit = mu_init, sigma_init
    
    # 拟合曲线用更细的网格
    x_fit = np.linspace(fu_data.min() - 0.5*sigma_fit, 
                        fu_data.max() + 0.5*sigma_fit, 500)
    pdf_fit = gaussian_pdf(x_fit, mu_fit, sigma_fit)
    
    # 计算CDF
    cdf_theoretical = stats.norm.cdf(x_fit, loc=mu_fit, scale=sigma_fit)
    
    # 经验CDF
    fu_sorted = np.sort(fu_data)
    empirical_cdf = np.arange(1, len(fu_sorted) + 1) / len(fu_sorted)
    
    # 理论CDF在数据点处的值
    cdf_at_data = stats.norm.cdf(fu_sorted, loc=mu_fit, scale=sigma_fit)
    
    return {
        'mu': mu_fit,
        'sigma': sigma_fit,
        'mu_std': mu_init,  # 经验均值
        'sigma_std': sigma_init,  # 经验标准差
        'fu_data': fu_data,
        'hist_counts': hist_counts,
        'hist_edges': hist_edges,
        'bin_centers': bin_centers,
        'x_fit': x_fit,
        'pdf_fit': pdf_fit,
        'cdf_theoretical': cdf_theoretical,
        'fu_sorted': fu_sorted,
        'empirical_cdf': empirical_cdf,
        'cdf_at_data': cdf_at_data,
    }


# ============================================================
# 5. 科研风格可视化
# ============================================================

def plot_fu_distribution(fit_result, save_path=None, figsize=(18, 10)):
    """
    绘制fu的高斯分布分析图（双Y轴科研风格）。
    
    参考图片风格：
      - 左Y轴: Probability density（直方图 + 高斯拟合曲线）
      - 右Y轴: Cumulative probability（经验CDF + 理论CDF）
      - X轴: Transition force fu
    
    参数:
        fit_result: fit_gaussian_to_fu()的输出
        save_path: 保存路径（不含扩展名）
        figsize: 图形尺寸
    """
    mu = fit_result['mu']
    sigma = fit_result['sigma']
    fu_data = fit_result['fu_data']
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # ---- 左Y轴: 概率密度 ----
    # 直方图
    n_bins = len(fit_result['hist_counts'])
    counts, edges, patches = ax1.hist(
        fu_data, bins=n_bins, density=True, 
        color='lightcoral', edgecolor='black', 
        alpha=0.5, linewidth=1.2, zorder=2,
        label='Histogram'
    )
    
    # 高斯拟合曲线
    ax1.plot(fit_result['x_fit'], fit_result['pdf_fit'],
             color='blue', linewidth=5, linestyle='-',
             label=f'Gaussian fit ($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})',
             zorder=5)
    
    ax1.set_xlabel(r'Transition force $f_u$', fontsize=35)
    ax1.set_ylabel('Probability density', fontsize=35, color='black')
    ax1.tick_params(axis='y', labelcolor='black', direction='in',
                    top=True, right=False, width=2, length=10)
    ax1.tick_params(axis='x', direction='in', top=True, bottom=True,
                    width=2, length=10)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='minor', direction='in',
                    top=True, right=False, width=1.5, length=5)
    
    # 左Y轴范围：从0到最大密度值+缓冲
    max_density = max(np.max(fit_result['pdf_fit']), np.max(counts) if len(counts) > 0 else 0)
    ax1.set_ylim(0, max_density * 1.15)
    
    # # ---- 右Y轴: 累积概率 ----
    # ax2 = ax1.twinx()
    
    # # 经验CDF（阶梯线）
    # ax2.plot(fit_result['fu_sorted'], fit_result['empirical_cdf'],
    #          color='green', linewidth=4, linestyle='--',
    #          label='Empirical CDF', zorder=4)
    
    # # 理论CDF（高斯分布的CDF）
    # ax2.plot(fit_result['x_fit'], fit_result['cdf_theoretical'],
    #          color='purple', linewidth=4, linestyle=':',
    #          label='Theoretical CDF', zorder=4)
    
    # ax2.set_ylabel('Cumulative probability', fontsize=35, color='black')
    # ax2.tick_params(axis='y', labelcolor='black', direction='in',
    #                 top=True, right=True, width=2, length=10)
    # ax2.set_ylim(0, 1.05)
    # ax2.minorticks_on()
    # ax2.tick_params(axis='y', which='minor', direction='in',
    #                 top=True, right=True, width=1.5, length=5)
    
    # ---- 图例 ----
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    # lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1,
               loc='upper left', fontsize=25,
               framealpha=0.9, edgecolor='gray',
               fancybox=True, shadow=False)
    
    # ---- 标题 ----
    ax1.set_title(r'$f_u$ Gaussian Distribution Analysis', 
                  fontsize=38, pad=20)
    
    # ---- 网格 ----
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1, zorder=0)
    
    # ---- 边框强化 ----
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    
    # ---- 自动X轴范围 ----
    x_min = fu_data.min() - 0.3 * sigma
    x_max = fu_data.max() + 0.3 * sigma
    ax1.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    # ---- 保存 ----
    if save_path:
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"图形已保存至: {save_path}.png")
    
    return fig, (ax1)


def plot_fu_by_chain(fu_by_chain, save_path=None, figsize=(16, 10)):
    """
    绘制每条链的fu分布散点图。
    """
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
         n_bins=50, save_dir=None):
    """
    主函数：读取C++输出 -> 提取fu -> 高斯拟合 -> 可视化
    
    参数:
        data_dir: C++程序输出CSV文件的目录路径
                  如果为None，则使用C++代码中的默认路径
        num_chains: 链的数量M
        num_domains: 每链domain数N
        n_bins: 直方图bin数
        save_dir: 结果保存目录
    """
    # 默认路径
    if data_dir is None:
        data_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Gibbs_Optimization_results/100_chains_IMS/10_100_C_file"
    
    if save_dir is None:
        save_dir = data_dir
    
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  M x N Domain链解折叠力(fu)统计分析")
    print("=" * 80)
    print(f"  数据目录: {data_dir}")
    print(f"  链数 M = {num_chains}")
    print(f"  Domain数 N = {num_domains}")
    print(f"  直方图bin数 = {n_bins}")
    print()
    
    # ---- Step 1: 加载数据 ----
    print("[Step 1/4] 加载所有链的CSV结果文件...")
    all_chains = load_all_chain_results(data_dir, num_chains, num_domains)
    print(f"  成功加载 {len(all_chains)} 条链")
    print()
    
    # ---- Step 2: 提取解折叠力fu ----
    print("[Step 2/4] 从f-r曲线中提取解折叠力fu...")
    fu_list, fu_by_chain = extract_all_unfolding_forces(all_chains, num_domains)
    print(f"  成功提取 {len(fu_list)} 个fu值")
    print(f"  fu范围: [{fu_list.min():.4f}, {fu_list.max():.4f}]")
    print(f"  fu均值: {fu_list.mean():.4f}")
    print(f"  fu标准差: {fu_list.std():.4f}")
    print()
    
    # 保存fu原始数据
    fu_save_path = save_dir / "unfolding_forces_raw.csv"
    np.savetxt(fu_save_path, fu_list, delimiter=',', 
               header='unfolding_force_fu', comments='')
    print(f"  fu原始数据已保存至: {fu_save_path}")
    print()
    
    # ---- Step 3: 高斯拟合 ----
    print("[Step 3/4] 对fu分布进行高斯拟合...")
    fit_result = fit_gaussian_to_fu(fu_list, n_bins=n_bins)
    print(f"  拟合结果:")
    print(f"    μ (均值)     = {fit_result['mu']:.6f}")
    print(f"    σ (标准差)   = {fit_result['sigma']:.6f}")
    print(f"    经验均值     = {fit_result['mu_std']:.6f}")
    print(f"    经验标准差   = {fit_result['sigma_std']:.6f}")
    print()
    
    # 保存拟合参数
    params_save_path = save_dir / "gaussian_fit_parameters.csv"
    with open(params_save_path, 'w') as f:
        f.write("Parameter,Value,Description\n")
        f.write(f"mu,{fit_result['mu']:.10f},高斯拟合均值\n")
        f.write(f"sigma,{fit_result['sigma']:.10f},高斯拟合标准差\n")
        f.write(f"mu_empirical,{fit_result['mu_std']:.10f},经验均值\n")
        f.write(f"sigma_empirical,{fit_result['sigma_std']:.10f},经验标准差\n")
        f.write(f"N_total,{len(fu_list)},总的fu数据点数\n")
        f.write(f"N_chains,{num_chains},链数\n")
        f.write(f"N_domains,{num_domains},每链domain数\n")
    print(f"  拟合参数已保存至: {params_save_path}")
    print()
    
    # ---- Step 4: 可视化 ----
    print("[Step 4/4] 生成科研风格可视化...")
    viz_save_path = save_dir / "fu_gaussian_distribution"
    fig, axes = plot_fu_distribution(fit_result, save_path=str(viz_save_path))
    
    # 额外：按链分布图
    viz_chain_path = save_dir / "fu_distribution_by_chain"
    fig2, ax2 = plot_fu_by_chain(fu_by_chain, save_path=str(viz_chain_path))
    
    print()
    print("=" * 80)
    print("  分析完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - {viz_save_path}.png     : fu高斯分布分析主图")
    print(f"  - {viz_chain_path}_by_chain.png : 各链fu分布散点图")
    print(f"  - {fu_save_path.name}           : fu原始数据")
    print(f"  - {params_save_path.name}       : 高斯拟合参数")
    
    return fit_result, fu_list, fu_by_chain

if __name__ == "__main__":
    main()