"""
模拟数据处理程序：绘制f-r和n-r曲线，比较三种：1.最小化自由能(red+gray) 2. 理论边界(purple) 3.连续化理论(Blue+black)
系统：100条10个domain串联的链
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
import sys

# ============ 字体设置 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
# 字体设置
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

# 字体大小
title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 25
legend_title_fontsize = 35

# 线宽和尺寸
axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 5
lines_markersize = 15

# 刻度方向
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

# 图形设置
figure_dpi = 100
savefig_dpi = 300

# ============ 应用全局设置 ============
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

# 基本参数
xi_f = 5.0  # 折叠态持续长度
alpha = 7.0      # 解折叠系数
E0 = 1.0     # 能量基准值
Ek = 4.0     # 能量系数
N = 10.0     # domain 的数量
k1 = 9.0
k2 = 1.36
R0 = 20.0    # 初始首末端距离

def energy_term_U(n_i, DeltaEi):
    """能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)"""
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i, xi_fi):
    """轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (ξ_ui - ξ_fi)"""
    xi_ui = alpha * xi_fi
    return xi_fi + n_i * (xi_ui - xi_fi)

def end_to_end_factor_x_i(r_i, n_i, xi_fi):
    """端到端因子: x_i(r_i, n_i) = r_i / L_{ci}(n_i)"""
    L_ci = contour_length_Lci(n_i, xi_fi)
    return r_i / L_ci

def WLC_free_energy(x_i, L_ci):
    """WLC自由能: F_{WLC}(x_i, n_i) = (1/4) L_{ci} * [x_i^2 (3 - 2x_i) / (1 - x_i)]"""
    if x_i >= 0.999:
        return float('inf')
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, xi_fi, f_ext=0.0):
    """单个domain的自由能: F_d(x_i, n_i) = F_{WLC}(x_i, n_i) + U(n_i) - f_ext * x_i * L_{ci}"""
    DeltaEi = E0 + Ek * (xi_fi - 5.0)
    L_ci = contour_length_Lci(n_i, xi_fi)
    x_i = end_to_end_factor_x_i(r_i, n_i, xi_fi)
    F_wlc = WLC_free_energy(x_i, L_ci)
    Ui = energy_term_U(n_i, DeltaEi)
    work_term = f_ext * x_i * L_ci
    return F_wlc + Ui - work_term

def Plot2state(r_fold, r_unfold, f, lineType):

    Et = single_domain_free_energy(r_unfold, 1.0, 10.0) - single_domain_free_energy(r_fold, 0.0, 10.0) 
    print(Et)
    p_u = 1 - 1/(1 + np.exp(-Et - f * (r_fold - r_unfold)))
    line = plt.plot(f, 10*p_u, lineType, color='blue', linewidth=lines_linewidth, label=f'$\Delta E_t = {Et:.1f}$, $r_s = {r_unfold - r_fold:.1f}$', zorder=3)
    
    return line

def MSforce(r, Lc):
    x = np.asarray(r, dtype=np.float64) / Lc
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def PlotBundaryAsy(f_min = 0.01, f_max = 10.0):
    f_val = np.linspace(f_min, f_max, 200)
    x_val = end_to_end_factor3(f_val)

    # 完全没打开
    r_val1 = x_val*N*xi_f
    # 全部打开
    r_val2 = x_val*N*alpha*xi_f

    # 绘图
    line1 = plt.plot(r_val1, f_val, "-", color='blue', linewidth=lines_linewidth, label=f'Fully folded $L_c = 50$', zorder=3)
    line2 = plt.plot(r_val2, f_val, "--", color='blue', linewidth=lines_linewidth, label=f'Fully unfolded $L_c = 350$', zorder=3)
    
    return line1, line2

def PlotBundaryMS():
    r_val = np.linspace(0, N*alpha*xi_f, 1000)
    f_val1 = MSforce(r_val, N*xi_f)
    f_val2 = MSforce(r_val, N*alpha*xi_f)

    # 绘图
    line1 = plt.plot(r_val, f_val1, "-", color='blue', linewidth=lines_linewidth, label=f'Fully folded $L_c = 50$', zorder=3)
    line2 = plt.plot(r_val, f_val2, "--", color='blue', linewidth=lines_linewidth, label=f'Fully unfolded $L_c = 350$', zorder=3)
    
    return line1, line2
    
def load_chain_data(chain_idx, data_dir):
    """加载单条链的r和n数据"""
    r_file = os.path.join(data_dir, f"chain_{chain_idx}_r_values_unified.csv")
    n_file = os.path.join(data_dir, f"chain_{chain_idx}_n_values_unified.csv")
    
    try:
        # 读取CSV文件
        df_r = pd.read_csv(r_file, header=0)
        df_n = pd.read_csv(n_file, header=0)
        
        # 提取力值f（第一列）
        f = df_r.iloc[:, 0].astype(float).values
        
        # 计算r_opt和n_opt（对每个domain求和）
        r_opt = df_r.iloc[:, 1:].astype(float).sum(axis=1).values
        n_opt = df_n.iloc[:, 1:].astype(float).sum(axis=1).values
        
        return f, r_opt, n_opt
    except FileNotFoundError:
        print(f"警告: 未找到链 {chain_idx} 的数据文件")
        return None, None, None
    except Exception as e:
        print(f"加载链 {chain_idx} 数据时出错: {e}")
        return None, None, None

def create_unified_grid(all_f_values, f_min=0.0, f_max=10.0, num_points=1000):
    """创建统一的力值网格"""
    # 收集所有力值点
    all_points = []
    for f_values in all_f_values:
        if f_values is not None:
            all_points.extend(f_values)
    
    # 去除重复并排序
    unique_points = np.unique(all_points)
    
    # 创建统一的网格
    # 使用所有链力值点的并集作为基础
    unified_grid = np.linspace(f_min, f_max, num_points)
    
    # 或者使用所有力值点的最小和最大值
    # if len(unique_points) > 0:
    #     f_min_actual = np.min(unique_points)
    #     f_max_actual = np.max(unique_points)
    #     unified_grid = np.linspace(f_min_actual, f_max_actual, num_points)
    
    return unified_grid

def interpolate_to_unified_grid(f_original, r_original, n_original, unified_f_grid):
    """将单条链的数据插值到统一网格上"""
    if f_original is None or len(f_original) < 2:
        return None, None
    
    # 只在内插范围内插值，不进行外推
    f_min, f_max = f_original[0], f_original[-1]
    mask = (unified_f_grid >= f_min) & (unified_f_grid <= f_max)
    
    if np.sum(mask) == 0:
        return None, None
    
    # 创建插值函数 - 使用原始数据顺序
    try:
        r_interp_func = interp1d(f_original, r_original, kind='linear', 
                                 bounds_error=False, fill_value=np.nan)
        n_interp_func = interp1d(f_original, n_original, kind='linear',
                                 bounds_error=False, fill_value=np.nan)
        
        # 在统一网格上插值
        r_interpolated = np.full_like(unified_f_grid, np.nan)
        n_interpolated = np.full_like(unified_f_grid, np.nan)
        
        r_interpolated[mask] = r_interp_func(unified_f_grid[mask])
        n_interpolated[mask] = n_interp_func(unified_f_grid[mask])
        
        return r_interpolated, n_interpolated
    except Exception as e:
        print(f"插值时出错: {e}")
        return None, None

def Lc(f):
    """
    物理含义为：施加外力f时的平均轮廓长度
    """
    contour_length =  N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))
    
    return contour_length

def end_to_end_factor1(f):
    """
    WLC的f(x)的近似反函数
    根据图中公式正确实现：
    x(f) = 4/3 - 4/(3√(f+1)) - 10e^{√[4]{900/f}} / [√f * (e^{√[4]{900/f}} - 1)^2] + f^1.62/(3.55 + 3.8f^2.2)
    """
    f = np.asarray(f, dtype=np.float64)
    
    # 创建结果数组
    x = np.zeros_like(f)
    
    # 只处理f>0的情况
    mask = f > 0
    if np.any(mask):
        f_masked = f[mask]
        
        # 第一项
        term1 = 4/3
        
        # 第二项：-4/(3√(f+1))
        term2 = -4/(3 * np.sqrt(f_masked + 1))
        
        # 第三项：-10e^{(900/f)^0.25} / [√f * (e^{(900/f)^0.25} - 1)^2]
        # 先计算指数部分
        exponent = (900 / f_masked) ** 0.25
        exp_term = np.exp(exponent)
        
        # 计算分母：√f * (exp_term - 1)^2
        denominator = np.sqrt(f_masked) * (exp_term - 1) ** 2
        
        # 避免分母为0
        denominator = np.where(denominator == 0, np.inf, denominator)
        
        # 第三项
        term3 = -10 * exp_term / denominator
        
        # 第四项：f^1.62/(3.55 + 3.8f^2.2)
        term4 = f_masked**1.62 / (3.55 + 3.8 * f_masked**2.2)
        
        # 总和
        x[mask] = term1 + term2 + term3 + term4
    
    # 确保x在合理范围内（对于WLC，x应在0-1之间）
    x = np.clip(x, 0, 1)
    
    return x

def end_to_end_factor2(f):
    """
    WLC的f(x)的近似反函数
    """
    a = 4/3*(2 + np.tanh(0.1*(f - 2)))
    x = 1 - 1/np.sqrt(a*f + 1)
    return x

def end_to_end_factor3(f):
    # Marko-Siggia的高力渐近形式
    x = 1 - 0.5*np.sqrt(1/f)
    return x

def PlotfTheory(f_min = 0.01, f_max = 10.0, lineType1 = '-', lineType2 = '--'):
    """f-拉伸曲线的解析公式——双曲正切"""
    f_val = np.linspace(f_min, f_max, 200)
    Length = Lc(f_val)

    #x_val1 = end_to_end_factor1(f_val)
    #x_val2 = end_to_end_factor2(f_val)
    x_val3 = end_to_end_factor3(f_val)
    
    #r_val1 = x_val1*Length
    #r_val2 = x_val2*Length
    r_val3 = x_val3*Length

    # line1 = plt.plot(r_val1, f_val, lineType1, color='purple', linewidth=lines_linewidth, label=f'Theory 1', zorder=2)
    # line2 = plt.plot(r_val2, f_val, lineType2, color='purple', linewidth=lines_linewidth, label=f'Theory', zorder=3)
    line3 = plt.plot(r_val3, f_val, lineType2, color='purple', linewidth=lines_linewidth, label=f'Theory', zorder=3)

    return line3

def PlotnTheory(f_min = 0.01, f_max = 10.0, lineType = '-'):
    """n-拉伸曲线的解析公式——双曲正切"""
    f_val = np.linspace(f_min, f_max, 200)
    ntheory = 0.5*(1.0 + np.tanh(k1*(f_val - k2)))
    line = plt.plot(f_val, ntheory, lineType, color='purple', linewidth=lines_linewidth, label=f'Theory', zorder=2)
    return line

def StressOptimization(R0, r_val, f_val):
    r_val = np.asarray(r_val)
    f_val = np.asarray(f_val)

    # 只保留 r >= R0 的数据点
    mask = r_val >= R0
    if not np.any(mask):
        raise ValueError("没有找到 r >= R0 的数据点，请检查 R0 或数据范围。")

    r_selected = r_val[mask]
    lambda_ = r_selected / R0                     # λ = r / R0
    r2 = lambda_ ** (-0.5) * R0                   # 对应的另一条链伸长

    # 线性插值获取力值，超出范围时使用边界值
    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])

    # 可选：检查外推警告
    if np.any(r_selected < r_val[0]) or np.any(r_selected > r_val[-1]):
        print("警告：某些 r1 值超出原始数据范围，使用了边界值。")
    if np.any(r2 < r_val[0]) or np.any(r2 > r_val[-1]):
        print("警告：某些 r2 值超出原始数据范围，使用了边界值。")

    # 计算应力：σ = R0 [ F'(λR0) - λ^{-3/2} F'(λ^{-1/2}R0) ]
    sigma = R0 * (f1 - lambda_ ** (-1.5) * f2)

    # 确保包含 λ=1 且 σ=0
    eps = 1e-12
    if len(lambda_) == 0:
        return lambda_, sigma   # 实际不会发生，因前面已检查非空

    # 检查第一个点（即最小 λ）是否接近 1
    if np.abs(lambda_[0] - 1.0) > eps:
        # 不包含 λ=1，插入 (1, 0) 到数组开头
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        # 已包含 λ≈1，将该点应力精确设为 0
        sigma[0] = 0.0

    return lambda_, sigma

def StressBoundry(R0):
    # 完全不打开极限
    Lc1 = N*xi_f
    r1_val = np.linspace(R0, 0.95*Lc1, 1000)
    lambda1 = r1_val/R0
    r1_prime = lambda1 ** (-0.5) * R0
    sigma1 = R0*(MSforce(r1_val, Lc1)- lambda1 ** (-1.5) * MSforce(r1_prime, Lc1))

    # 全部都打开极限
    Lc2 = alpha*N*xi_f
    r2_val = np.linspace(R0, 0.95*Lc2, 1000)
    lambda2 = r2_val/R0
    r2_prime = lambda2 ** (-0.5) * R0
    sigma2 = R0*(MSforce(r2_val, Lc2)- lambda2 ** (-1.5) * MSforce(r2_prime, Lc2))

    return lambda1, sigma1, lambda2, sigma2

def process_all_chains(data_dir, num_chains=100):
    """处理所有链的数据"""
    print(f"正在处理 {num_chains} 条链的数据...")
    
    all_f_values = []
    all_r_values = []
    all_n_values = []
    
    for chain_idx in range(1, num_chains + 1):
        f, r_opt, n_opt = load_chain_data(chain_idx, data_dir)
        
        if f is not None and r_opt is not None and n_opt is not None:
            all_f_values.append(f)
            all_r_values.append(r_opt)
            all_n_values.append(n_opt)
            
            if chain_idx % 10 == 0:
                print(f"  已加载 {chain_idx} 条链")
    
    print(f"成功加载 {len(all_f_values)} 条链的数据")
    return all_f_values, all_r_values, all_n_values

def calculate_average_curves(unified_f_grid, all_r_interpolated, all_n_interpolated):
    """计算平均曲线"""
    # 转换为numpy数组
    r_matrix = np.array([r for r in all_r_interpolated if r is not None])
    n_matrix = np.array([n for n in all_n_interpolated if n is not None])
    
    if len(r_matrix) == 0:
        return None, None
    
    # 计算平均值（忽略NaN值）
    r_mean = np.nanmean(r_matrix, axis=0)
    n_mean = np.nanmean(n_matrix, axis=0)
    
    # 计算标准差（可选，用于绘制误差带）
    r_std = np.nanstd(r_matrix, axis=0)
    n_std = np.nanstd(n_matrix, axis=0)
    
    return r_mean, n_mean, r_std, n_std

def create_visualization(all_f_values, all_r_values, all_n_values,
                         unified_f_grid, r_mean, n_mean, r_std=None, n_std=None,
                         save_dir=None):
    """创建可视化图表"""
    
    # ============ 创建第一幅图: f-r_opt ============
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    
    # 绘制Marko-Siggia边界线_渐近
    # PlotBundaryAsy()
    # 绘制Marko-Siggia边界线_精确
    PlotBundaryMS()
    
    # 绘制所有链的原始轨迹（半透明灰色）
    for f, r_opt in zip(all_f_values, all_r_values):
        # 使用原始数据顺序，不进行排序
        ax1.plot(r_opt, f, 
                color='gray', alpha=0.5, linewidth=0.5, zorder=1)
    

    # 绘制平均曲线（红色）
    # 移除平均曲线中的NaN值
    valid_mask = ~np.isnan(r_mean)
    if np.any(valid_mask):
        ax1.plot(r_mean[valid_mask], unified_f_grid[valid_mask], 
                color='red', linewidth=lines_linewidth, 
                label='Average', zorder=2)
    
    # 绘制理论曲线
    PlotfTheory()

    # 设置标签和标题
    ax1.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax1.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax1.set_title(f'{len(all_f_values)} Chains: Force-extension Curve', 
                  fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax1.set_xlim(0.0, alpha*N*xi_f)
    ax1.set_ylim(0.0, 4.0)
    
    # 设置刻度参数
    ax1.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax1.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax1.minorticks_on()
    
    # 强化边框
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    # 保存第一幅图
    if save_dir:
        save_path1 = os.path.join(save_dir, 'Optmization vs. Theory.png')
        fig1.savefig(save_path1, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"f-r图已保存至: {save_path1}")
    
    # ============ 创建第二幅图: n_opt-f ============
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 9))

    # 绘制所有链的原始轨迹（半透明灰色）
    for f, n_opt in zip(all_f_values, all_n_values):
        # 使用原始数据顺序，不进行排序
        ax2.plot(f, n_opt/N, 
                color='gray', alpha=0.5, linewidth=0.5, zorder=1)
    
    # 绘制平均曲线（红色）
    valid_mask = ~np.isnan(n_mean)
    if np.any(valid_mask):
        ax2.plot(unified_f_grid[valid_mask], n_mean[valid_mask]/N, 
                color='red', linewidth=lines_linewidth, 
                label='Average', zorder=2)
        
    # 绘制理论曲线
    PlotnTheory()

    # 设置标签和标题
    ax2.set_xlabel('Force $f$', fontsize=label_fontsize)
    ax2.set_ylabel('Unfolded Fraction $n/N$', fontsize=label_fontsize)
    ax2.set_title(f'{len(all_f_values)} Chains: $n/N$ vs. $f$', 
                  fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax2.set_xlim(0.0, 3.0)
    ax2.set_ylim(-0.1, 1.1)
    
    # 设置刻度参数
    ax2.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax2.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax2.minorticks_on()
    
    # 强化边框
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    # 保存第二幅图
    if save_dir:
        save_path2 = os.path.join(save_dir, 'n_opt_f vs. theory.png')
        fig2.savefig(save_path2, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"n-f图已保存至: {save_path2}")


    # ============ 创建第三幅图: 3-chain model的本构曲线 ============
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 9))

    # 数值曲线：红色
    valid_mask = ~np.isnan(n_mean)
    if np.any(valid_mask):
        lambda_, sigma = StressOptimization(R0, r_mean[valid_mask], unified_f_grid[valid_mask])
    ax3.plot(lambda_, sigma, color='red', linewidth=lines_linewidth, label='Optimization', zorder=2)

    # 绘制边界：蓝色
    lambda1, sigma1, lambda2, sigma2 = StressBoundry(R0)
    ax3.plot(lambda1, sigma1, '-', color='blue', linewidth=lines_linewidth, label='Fully folded', zorder=1)
    ax3.plot(lambda2, sigma2, '--', color='blue', linewidth=lines_linewidth, label='Fully unfolded', zorder=2)

    # 连续化的理论曲线：紫色
    f_vals = np.linspace(0.01, 20.0, 1000)
    r_vals = end_to_end_factor3(f_vals)*Lc(f_vals)
    clambda, csigma = StressOptimization(R0, r_vals, f_vals)
    ax3.plot(clambda, csigma, '-', color='purple', linewidth=lines_linewidth, label='Theory', zorder=3)

    # 设置标签和标题
    ax3.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax3.set_ylabel('Stress $\sigma/\\rho k_B T$', fontsize=label_fontsize)
    ax3.set_title(f'Constitutive curve', 
                  fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax3.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax3.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax3.set_xlim(1.0, alpha*N*xi_f/R0)
    ax3.set_ylim(0.0, 100.0)
    
    # 设置刻度参数
    ax3.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax3.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax3.minorticks_on()
    
    # 强化边框
    for spine in ax3.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    # 保存第三幅图
    if save_dir:
        save_path3 = os.path.join(save_dir, 'Stress-strain.png')
        fig3.savefig(save_path3, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"本构曲线已保存至: {save_path3}")

    
    return fig1, fig2, fig3

def save_average_data(unified_f_grid, r_mean, n_mean, r_std=None, n_std=None, save_dir=None):
    """保存平均数据到CSV文件"""
    if save_dir is None:
        return
    
    # 创建DataFrame
    data_dict = {'Force_f': unified_f_grid, 'Average_r': r_mean, 'Average_n': n_mean}
    
    if r_std is not None:
        data_dict['Std_r'] = r_std
    if n_std is not None:
        data_dict['Std_n'] = n_std
    
    df = pd.DataFrame(data_dict)
    
    # 保存到CSV
    save_path = os.path.join(save_dir, 'average_curves.csv')
    df.to_csv(save_path, index=False)
    print(f"平均曲线数据已保存至: {save_path}")
    
    return df

def main():
    """主程序"""
    print("=" * 80)
    print("100条链数据处理和可视化程序")
    print("=" * 80)
    
    # ============ 在这里指定文件路径 ============
    data_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Gibbs_Optimization_results/100_chains_IMS/10_100_C_file"
    output_dir = data_dir  # 保存结果的目录
    num_chains = 100
    
    # 如果没有命令行参数，使用默认路径
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        output_dir = data_dir
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    if len(sys.argv) > 3:
        num_chains = int(sys.argv[3])
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"链数量: {num_chains}")
    
    # 处理所有链的数据
    all_f_values, all_r_values, all_n_values = process_all_chains(data_dir, num_chains)
    
    if len(all_f_values) == 0:
        print("未找到任何链的数据，程序退出")
        return
    
    # 创建统一的力值网格
    unified_f_grid = create_unified_grid(all_f_values, f_min=0.0, f_max=10.0, num_points=1000)
    print(f"统一力值网格点数: {len(unified_f_grid)}")
    
    # 将所有链的数据插值到统一网格上
    print("正在将所有链的数据插值到统一网格...")
    all_r_interpolated = []
    all_n_interpolated = []
    
    for f, r_opt, n_opt in zip(all_f_values, all_r_values, all_n_values):
        r_interp, n_interp = interpolate_to_unified_grid(f, r_opt, n_opt, unified_f_grid)
        if r_interp is not None and n_interp is not None:
            all_r_interpolated.append(r_interp)
            all_n_interpolated.append(n_interp)
    
    print(f"成功插值 {len(all_r_interpolated)} 条链的数据")
    
    # 计算平均曲线
    print("正在计算平均曲线...")
    r_mean, n_mean, r_std, n_std = calculate_average_curves(unified_f_grid, all_r_interpolated, all_n_interpolated)
    
    if r_mean is None or n_mean is None:
        print("无法计算平均曲线，程序退出")
        return
    
    # 保存平均数据
    save_average_data(unified_f_grid, r_mean, n_mean, r_std, n_std, output_dir)
    
    # 创建可视化图表
    print("\n" + "=" * 80)
    print("创建可视化图表...")
    print("=" * 80)
    
    fig1, fig2, fig3 = create_visualization(all_f_values, all_r_values, all_n_values,
                                      unified_f_grid, r_mean, n_mean, r_std, n_std,
                                      save_dir=output_dir)
    
    print("\n" + "=" * 80)
    print("程序执行完毕!")
    print(f"所有图表已保存至: {os.path.abspath(output_dir)}")
    print("=" * 80)
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"  处理链数: {len(all_f_values)}")
    print(f"  统一网格点数: {len(unified_f_grid)}")
    
    # 计算有效数据点比例
    valid_r_ratio = np.mean(~np.isnan(r_mean)) * 100
    valid_n_ratio = np.mean(~np.isnan(n_mean)) * 100
    print(f"  r平均曲线有效数据比例: {valid_r_ratio:.1f}%")
    print(f"  n平均曲线有效数据比例: {valid_n_ratio:.1f}%")
    
    # 显示平均曲线的范围
    r_valid = r_mean[~np.isnan(r_mean)]
    n_valid = n_mean[~np.isnan(n_mean)]
    if len(r_valid) > 0:
        print(f"  平均r范围: [{r_valid.min():.2f}, {r_valid.max():.2f}]")
    if len(n_valid) > 0:
        print(f"  平均n范围: [{n_valid.min():.2f}, {n_valid.max():.2f}]")

# ============ 运行主程序 ============
if __name__ == "__main__":
    main()