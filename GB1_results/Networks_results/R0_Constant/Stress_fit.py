"""
GB1数据处理完整脚本（修正版）
修复了读取Excel时数据类型导致的TypeError问题
统一由全局变量 strain_max 控制优化与可视化的拉伸比上限
拟合初始猜测、边界等从 main 函数传入
最终拟合参数保存为 fitting_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os

# ============================================================================
# 1. 全局样式与字体设置
# ============================================================================

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 30
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 3
lines_markersize = 35

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_right = True

figure_dpi = 100
savefig_dpi = 300

# 应用全局设置
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

# ============================================================================
# 2. 物理常数与核心函数定义
# ============================================================================

# 物理参数
xi_f = 3.6          # 折叠态持续长度
alpha = 7.6         # 解折叠系数
k1 = 6.5
k2 = 1.48

# ========== 全局控制参数：拉伸比上限 ==========
strain_max = 2.3    # 修改此值即可同时改变插值、拟合和绘图范围

def Lc(f, N):
    """
    计算轮廓长度 Lc(f)
    """
    return N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))

def MSforce(x):
    """
    Marko-Siggia 力-伸长关系
    """
    # 防止除以0，限制x上限
    x = np.clip(x, 0, 0.9999)
    return 0.25 * ((1 - x) ** (-2) - 1 + 4 * x)

def StressOptimization(R0, r_val, f_val):
    """
    根据单链f-r关系计算本构关系 sigma - lambda
    """
    r_val = np.asarray(r_val, dtype=float) # 确保类型
    f_val = np.asarray(f_val, dtype=float) # 确保类型

    mask = r_val >= R0
    if not np.any(mask):
        return np.array([1.0]), np.array([0.0])

    r_selected = r_val[mask]
    lambda_ = r_selected / R0
    r2 = lambda_ ** (-0.5) * R0

    # 插值获取对应的力
    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])

    sigma = R0 * (f1 - lambda_ ** (-1.5) * f2)

    # 确保从 lambda=1 开始
    if np.abs(lambda_[0] - 1.0) > 1e-12:
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        sigma[0] = 0.0
    
    return lambda_, sigma

def theory_constitutive(R0, N, lambda_max):
    """
    生成理论本构曲线数据
    """
    # Step 1-3: 生成 f-r 关系
    x = np.linspace(0.0, 0.99, 500)
    f = MSforce(x)
    Lc_vals = Lc(f, N)
    r = x * Lc_vals
    
    # Step 4-5: 计算本构关系
    lambda_theory, sigma_theory = StressOptimization(R0, r, f)
    
    # 截断至 lambda_max
    mask = lambda_theory <= lambda_max
    return lambda_theory[mask], sigma_theory[mask]

# ============================================================================
# 3. 数据处理函数 (修正部分)
# ============================================================================

def read_excel_data(file_path):
    """
    读取Excel数据，应变+1，返回字典
    修正：强制转换数据类型，防止TypeError
    """
    print(f"读取文件: {file_path}")
    xls = pd.ExcelFile(file_path)
    data_dict = {}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        n_cols = df.shape[1]
        n_groups = n_cols // 2
        groups_data = []
        
        for i in range(n_groups):
            # 使用 pd.to_numeric 强制转换为数值，errors='coerce' 将非数值转为NaN
            strain_series = pd.to_numeric(df.iloc[:, 2*i], errors='coerce')
            stress_series = pd.to_numeric(df.iloc[:, 2*i + 1], errors='coerce')
            
            strain = strain_series.values
            stress = stress_series.values
            
            # 移除NaN (现在可以安全使用np.isnan)
            mask = ~(np.isnan(strain) | np.isnan(stress))
            strain = strain[mask]
            stress = stress[mask]
            
            # 核心步骤：应变 + 1
            strain = strain + 1.0
            groups_data.append((strain, stress))
            
        data_dict[sheet_name] = groups_data
        print(f"  Sheet '{sheet_name}': 读取 {n_groups} 组数据")
        
    return data_dict

def interpolate_and_average(groups_data, strain_max_val, n_points=1000):
    """
    插值并对多组数据求平均
    """
    strain_grid = np.linspace(1.0, strain_max_val, n_points)
    stress_interpolated = []
    
    for strain, stress in groups_data:
        # 分段线性插值
        f_interp = interp1d(strain, stress, kind='linear', 
                           bounds_error=False, fill_value=np.nan)
        stress_interp = f_interp(strain_grid)
        stress_interpolated.append(stress_interp)
    
    stress_interpolated = np.array(stress_interpolated)
    # 计算平均值 (忽略NaN)
    stress_avg = np.nanmean(stress_interpolated, axis=0)
    
    return strain_grid, stress_avg

# ============================================================================
# 4. 拟合与优化
# ============================================================================

def fitting_objective(params, lambda_exp, sigma_exp, N, lambda_max):
    """
    目标函数：计算实验数据与理论曲线的误差
    params: [R0, G]
    """
    R0, G = params
    
    try:
        # 生成理论曲线
        lambda_theo, sigma_theo = theory_constitutive(R0, N, lambda_max)
        
        if len(lambda_theo) < 2:
            return 1e10
        
        # 将理论曲线插值到实验网格上
        interp_func = interp1d(lambda_theo, sigma_theo, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        sigma_theo_interp = interp_func(lambda_exp)
        
        # 相对误差平方和
        mask = ~np.isnan(sigma_exp) & ~np.isnan(sigma_theo_interp)
        error = np.sum(((sigma_exp[mask] - G * sigma_theo_interp[mask]) / (sigma_exp[mask] + 1e-6))**2)
        return error
    
    except Exception as e:
        return 1e10

def fit_parameters(lambda_exp, sigma_exp, N, lambda_max,
                   x0, bounds, options=None):
    """
    执行拟合优化
    参数:
        lambda_exp, sigma_exp: 实验数据
        N: 结构域数量
        lambda_max: 拉伸比上限
        x0: 初始猜测 [R0_init, G_init]
        bounds: 参数边界 [(R0_low, R0_high), (G_low, G_high)]
        options: 传递给 minimize 的选项字典 (可选)
    返回:
        R0_opt, G_opt
    """
    print(f"  拟合 N={N}...")
    if options is None:
        options = {'maxiter': 200}
    
    result = minimize(fitting_objective, 
                     x0=x0,
                     args=(lambda_exp, sigma_exp, N, lambda_max),
                     method='L-BFGS-B',
                     bounds=bounds,
                     options=options)
    
    # 检查收敛性
    if not result.success:
        print(f"    警告：拟合可能未收敛 (N={N})，信息: {result.message}")
    return result.x[0], result.x[1]

# ============================================================================
# 5. 可视化与保存
# ============================================================================

def save_networks_data(results_dict, save_dir):
    """
    保存平均拉伸曲线数据
    """
    data_to_save = {}
    for sheet_name, data in results_dict.items():
        N = data['N']
        data_to_save[f'lambda_N{N}'] = data['strain_avg']
        data_to_save[f'sigma_N{N}'] = data['stress_avg']
    
    df = pd.DataFrame(data_to_save)

    save_path = os.path.join(save_dir, 'networks_data.csv')
    df.to_csv(save_path, index=False)
    print(f"\n数据已保存: {save_path}")

def plot_fitting_comparison(results_dict, R0_opt_dict, G_opt_dict, lambda_max, save_dir):
    """
    绘制实验数据与理论曲线的对比图
    （x轴范围现在由全局 strain_max 控制）
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝、橙、绿
    markers = ['o', 's', '^']
    
    for idx, (sheet_name, data) in enumerate(results_dict.items()):
        N = data['N']
        lambda_exp = data['strain_avg']
        sigma_exp = data['stress_avg']
        
        R0 = R0_opt_dict[sheet_name]
        G = G_opt_dict[sheet_name]
        
        # 生成优化后的理论曲线
        lambda_th, sigma_th = theory_constitutive(R0, N, lambda_max)
        
        # 绘制实验点 (散点) —— 添加黑色边界，填充颜色为各曲线颜色
        ax.plot(lambda_exp, sigma_exp, markers[idx],
               markerfacecolor=colors[idx],    # 填充颜色
               markeredgecolor='black',        # 黑色边界
               markeredgewidth=0.1,            # 边界宽度
               markersize=6,                   # 点的大小
               alpha=0.8,                      # 透明度（仅影响填充）
               linestyle='None',               # 只绘制散点，无连线
               label=f'N={N} (Exp)')
        
        # 绘制理论线 (实线)
        ax.plot(lambda_th, G * sigma_th, '-',
               color=colors[idx], linewidth=lines_linewidth,
               label=f'N={N} (Fit, $R_0$={R0:.1f})')
    
    ax.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Stress $\sigma$  [kPa]', fontsize=label_fontsize)
    ax.set_title('Constitutive curve fitting', fontsize=title_fontsize, pad=20)
    
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=0.8*legend_fontsize, loc='best')
    
    # 使用全局参数控制横轴范围
    ax.set_xlim(1.0, lambda_max)
    ax.set_ylim(0, None)
    
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    ax.minorticks_on()
    
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'Constitutive_fit.png')
    plt.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"图表已保存: {save_path}")
    plt.close()

# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    print("="*80)
    print("GB1 数据处理任务开始")
    print("="*80)
    
    # 1. 读取数据
    save_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_Constant'
    file_path = os.path.join(save_dir,'GB1_Pull.xlsx')
    
    if not os.path.exists(file_path):
        print("警告: 文件不存在，请检查路径。")
        return
    
    data_dict = read_excel_data(file_path)
    
    # 2. 处理数据
    results_dict = {}
    N_map = {'(GB1)2': 2, '(GB1)4': 4, '(GB1)8': 8}
    
    for sheet_name, groups_data in data_dict.items():
        N = N_map.get(sheet_name, 2)
        strain_avg, stress_avg = interpolate_and_average(groups_data, strain_max)
        
        results_dict[sheet_name] = {
            'N': N,
            'strain_avg': strain_avg,
            'stress_avg': stress_avg
        }
    
    # 保存处理后的数据
    save_networks_data(results_dict, save_dir)
    
    # 3. 拟合参数
    # ====== 在此处定义初始猜测、边界和优化选项 ======
    # 拟合参数: R0 和 G
    bounds = [(1.0, 100.0),    # R0 的下界与上界
              (0.1, 100.0)]    # G 的下界与上界
    
    options = {'maxiter': 200}  # 优化器选项
    
    R0_opt_dict = {}
    G_opt_dict = {}
    
    print("\n开始拟合参数...")
    for sheet_name, data in results_dict.items():
        N = data['N']
        lambda_exp = data['strain_avg']
        sigma_exp = data['stress_avg']
        
        # 根据 N 构造初始猜测 (这里沿用之前的经验公式)
        R0_guess = 10.0 * np.sqrt(N)
        G_guess = 10.0
        x0 = [R0_guess, G_guess]
        
        # 调用拟合函数，传入所有外部定义的参数
        R0_opt, G_opt = fit_parameters(lambda_exp, sigma_exp, N, 
                                       lambda_max=strain_max,
                                       x0=x0,
                                       bounds=bounds,
                                       options=options)
        
        R0_opt_dict[sheet_name] = R0_opt
        G_opt_dict[sheet_name] = G_opt
    
    # 4. 保存拟合结果到 fitting_results.csv
    print("\n保存拟合结果到 fitting_results.csv ...")
    results_rows = []
    # 系统参数行
    sys_row = {
        'Parameter': 'system',
        'xi_f': xi_f,
        'alpha': alpha,
        'k1': k1,
        'k2': k2,
        'strain_max': strain_max,
        'R0_bound_lower': bounds[0][0],
        'R0_bound_upper': bounds[0][1],
        'G_bound_lower': bounds[1][0],
        'G_bound_upper': bounds[1][1],
        'N': None,
        'R0': None,
        'G': None
    }
    results_rows.append(sys_row)
    # 各拟合参数行
    for sheet_name, data in results_dict.items():
        N = data['N']
        row = {
            'Parameter': f'N={N}',
            'xi_f': None,
            'alpha': None,
            'k1': None,
            'k2': None,
            'strain_max': None,
            'R0_bound_lower': None,
            'R0_bound_upper': None,
            'G_bound_lower': None,
            'G_bound_upper': None,
            'N': N,
            'R0': R0_opt_dict[sheet_name],
            'G': G_opt_dict[sheet_name]
        }
        results_rows.append(row)
    
    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(os.path.join(save_dir, 'fitting_results.csv'), index=False)
    print("拟合结果已保存。")

    # 5. 绘制对比图
    plot_fitting_comparison(results_dict, R0_opt_dict, G_opt_dict, 
                           lambda_max=strain_max,
                           save_dir=save_dir)
    
    print("="*80)
    print("\n任务完成")
    print("="*80)

if __name__ == "__main__":
    main()