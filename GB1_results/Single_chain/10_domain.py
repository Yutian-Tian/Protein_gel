"""
基于吉布斯自由能的优化 - 自适应力值细化多domain版本
系统：N个独立的domain
"""

import numpy as np
import pandas as pd
import os

# 参数设置
xi_f = 3.6
alpha = 7.6   # 解折叠系数

E_mean = 11.9 
E_std = 1.7  # 能量系数
E_delta = 5.0

N = 10    # domain的数量
r_grid = 100  # r的网格数
n_grid = 10   # n的网格数
f_grid_initial = 100  # 初始力值网格数
f_max = 10.0  # 扫描力f的最大值

# 自适应细化参数
refinement_threshold = 0.5  # n变化超过此阈值时进行细化
max_refinement_level = 10   # 最大细化层级
tolerance = 1e-6            # 最小f间隔

# 上下界设置（用于生成xi_f）
upper_bound = E_mean + E_delta
lower_bound = E_mean - E_delta

# 设置存储路径
save_path = "/home/tyt/project/protein_gel/GB1_results/Single_chain/results"
os.makedirs(save_path, exist_ok=True)

def energy_term_U(n_i, DeltaEi):
    """
    能量项: U(n_i) = ΔE_i n_i - ΔE_i cos(2π n_i)
    """
    return DeltaEi * n_i - DeltaEi * np.cos(2 * np.pi * n_i)

def contour_length_Lci(n_i):
    """
    轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (ξ_ui - ξ_fi)
    """
    return xi_f + n_i * (alpha - 1.0)*xi_f

def end_to_end_factor_x_i(r_i, n_i):
    """
    端到端因子: x_i(r_i, n_i) = r_i / L_{ci}(n_i)
    """
    L_ci = contour_length_Lci(n_i)
    return r_i / L_ci

def WLC_free_energy(x_i, L_ci):
    """
    WLC自由能: F_{WLC}(x_i, n_i) = (1/4) L_{ci} * [x_i^2 (3 - 2x_i) / (1 - x_i)]
    """
    # 避免除以零，当x_i接近1时
    if x_i >= 0.999:
        return float('inf')
    
    return 0.25 * L_ci * (x_i**2 * (3.0 - 2.0 * x_i) / (1.0 - x_i))

def single_domain_free_energy(r_i, n_i, DeltaEi, f_ext=0.0):
    """
    单个domain的自由能: F_d(x_i, n_i) = F_{WLC}(x_i, n_i) + U(n_i) - f_ext * x_i * L_{ci}
    """
    # 计算轮廓长度
    L_ci = contour_length_Lci(n_i)
    
    x_i = end_to_end_factor_x_i(r_i, n_i)
    
    # 计算WLC自由能
    F_wlc = WLC_free_energy(x_i, L_ci)
    
    # 计算能量项
    Ui = energy_term_U(n_i, DeltaEi)
    
    # 计算外力做功项
    work_term = f_ext * x_i * L_ci
    
    # 总自由能
    F_di = F_wlc + Ui - work_term
    
    return F_di

def optimize_single_point(f, DeltaEi, opt_tolerance=1e-8):
    """
    优化单个力值点对应的自由能
    使用局部网格搜索方法
    """
    r_max_initial = contour_length_Lci(1.0)
    
    # 初始搜索范围
    r_search_min, r_search_max = 0.0, r_max_initial
    n_search_min, n_search_max = 0.0, 1.0
    
    # 初始网格步长
    r_step = (r_search_max - r_search_min) / (r_grid - 1)
    n_step = (n_search_max - n_search_min) / (n_grid - 1)
    
    # 初始化最优值
    min_Fd = float('inf')
    r_best, n_best = 0.0, 0.0
    
    # 迭代细化
    while True:
        # 生成当前搜索网格
        r_vals = np.linspace(r_search_min, r_search_max, r_grid)
        n_vals = np.linspace(n_search_min, n_search_max, n_grid)
        
        # 网格搜索
        for r in r_vals:
            for n in n_vals:
                Fd = single_domain_free_energy(r, n, DeltaEi, f_ext=f)
                if Fd < min_Fd:
                    min_Fd = Fd
                    r_best, n_best = r, n
        
        # 检查收敛
        if r_step <= opt_tolerance and n_step <= opt_tolerance:
            break
        
        # 缩小搜索范围
        r_search_min = max(0.0, r_best - r_step)
        r_search_max = min(r_max_initial, r_best + r_step)
        n_search_min = max(0.0, n_best - n_step)
        n_search_max = min(1.0, n_best + n_step)
        
        # 更新步长
        r_step = (r_search_max - r_search_min) / 10.0
        n_step = (n_search_max - n_search_min) / 10.0
    
    x_best = end_to_end_factor_x_i(r_best, n_best)
    return r_best, n_best, min_Fd, x_best

def detect_transition_regions(f_vals, n_opt, threshold=0.5):
    """检测n_opt发生显著变化的区域"""
    n = len(f_vals)
    transition_regions = []
    
    for i in range(n-1):
        dn = n_opt[i+1] - n_opt[i]
        if abs(dn) >= threshold:
            current_start = f_vals[i]
            current_end = f_vals[i+1]
            transition_regions.append(np.array([current_start, current_end]))
    
    return np.array(transition_regions) if transition_regions else np.array([])

def refine_force_grid(f_vals, n_opt, refinement_threshold=0.5, tolerance=1e-10):
    """在n_opt变化剧烈的区域细化力值网格"""
    # 初始网格
    refined_f = f_vals.copy()
    
    # 检测过渡区域
    transition_regions = detect_transition_regions(refined_f, n_opt, refinement_threshold)
    
    if len(transition_regions) == 0:
        return refined_f
    
    # 收集需要细化的区间
    new_f_points = []
    
    for i in range(len(transition_regions)):
        f_start = transition_regions[i, 0]
        f_end = transition_regions[i, 1]
        
        # 在当前区间内插入中点
        f_mid = (f_start + f_end) / 2
        
        # 跳过已经存在的点
        if not any(abs(f_mid - f) < tolerance for f in refined_f):
            new_f_points.append(f_mid)
    
    if not new_f_points:
        return refined_f
    
    # 合并结果
    for f in new_f_points:
        refined_f = np.append(refined_f, f)
    
    # 按f值排序
    refined_f = np.sort(refined_f)
    
    return refined_f

def optimize_single_domain_adaptive(DeltaEi, domain_idx):
    """
    使用自适应力值细化优化单个domain
    返回：力值网格，最优r, n, Fd, x
    """
    print(f"开始优化 domain {domain_idx+1} ($\Delta E$ = {DeltaEi:.2f})...")
    
    # 初始均匀力值网格
    f_vals_current = np.linspace(0.0, f_max, f_grid_initial)
    
    # 初始优化
    r_opt_current = []
    n_opt_current = []
    Fd_min_current = []
    x_opt_current = []
    
    for i, f in enumerate(f_vals_current):
        r_opt, n_opt, Fd_min, x_opt = optimize_single_point(f, DeltaEi)
        r_opt_current.append(r_opt)
        n_opt_current.append(n_opt)
        Fd_min_current.append(Fd_min)
        x_opt_current.append(x_opt)
    
    r_opt_current = np.array(r_opt_current)
    n_opt_current = np.array(n_opt_current)
    Fd_min_current = np.array(Fd_min_current)
    x_opt_current = np.array(x_opt_current)
    
    print(f"  Domain {domain_idx+1} 初始优化完成")
    
    # 自适应细化循环
    for level in range(max_refinement_level):
        # 检测并细化过渡区域
        refined_f_vals = refine_force_grid(f_vals_current, n_opt_current, refinement_threshold)
        
        # 检查是否需要细化
        if len(refined_f_vals) == len(f_vals_current):
            print(f"  Domain {domain_idx+1} 没有检测到需要细化的区域，停止细化")
            break
        
        # 检查精度是否满足要求
        if len(refined_f_vals) > 1:
            f_diffs = np.diff(np.sort(refined_f_vals))
            min_f_interval = np.min(f_diffs)
            
            if min_f_interval <= tolerance:
                print(f"  Domain {domain_idx+1} 精度已达到要求，停止细化")
                break
        
        # 识别新增的力值点
        new_f_points = []
        for f in refined_f_vals:
            is_new = True
            for existing_f in f_vals_current:
                if abs(f - existing_f) < 1e-10:
                    is_new = False
                    break
            if is_new:
                new_f_points.append(f)
        
        if not new_f_points:
            print(f"  Domain {domain_idx+1} 没有新增点，停止细化")
            break
        
        
        # 优化新增点
        new_r_opt = []
        new_n_opt = []
        new_Fd_min = []
        new_x_opt = []
        
        for f in new_f_points:
            r_opt, n_opt, Fd_min, x_opt = optimize_single_point(f, DeltaEi)
            new_r_opt.append(r_opt)
            new_n_opt.append(n_opt)
            new_Fd_min.append(Fd_min)
            new_x_opt.append(x_opt)
        
        # 创建从力值到结果的映射字典
        result_dict = {}
        for i, f in enumerate(f_vals_current):
            result_dict[f] = {
                'r': r_opt_current[i],
                'n': n_opt_current[i],
                'Fd': Fd_min_current[i],
                'x': x_opt_current[i]
            }
        
        # 添加新点的结果
        for i, f in enumerate(new_f_points):
            result_dict[f] = {
                'r': new_r_opt[i],
                'n': new_n_opt[i],
                'Fd': new_Fd_min[i],
                'x': new_x_opt[i]
            }
        
        # 按照refined_f_vals的顺序构建最终结果数组
        f_vals_final = refined_f_vals.copy()
        r_opt_final = []
        n_opt_final = []
        Fd_min_final = []
        x_opt_final = []
        
        for f in f_vals_final:
            r_opt_final.append(result_dict[f]['r'])
            n_opt_final.append(result_dict[f]['n'])
            Fd_min_final.append(result_dict[f]['Fd'])
            x_opt_final.append(result_dict[f]['x'])
        
        # 更新当前结果
        f_vals_current = np.array(f_vals_final)
        r_opt_current = np.array(r_opt_final)
        n_opt_current = np.array(n_opt_final)
        Fd_min_current = np.array(Fd_min_final)
        x_opt_current = np.array(x_opt_final)
        
        # 检查是否满足精度要求
        if len(f_vals_current) > 1:
            f_diffs = np.diff(f_vals_current)
            min_f_interval = np.min(f_diffs)
            
            if min_f_interval <= tolerance:
                print(f"  Domain {domain_idx+1} 精度已达到要求，停止细化")
                break
    
    print(f"  Domain {domain_idx+1} 优化完成，总点数: {len(f_vals_current)}")
    
    return f_vals_current, r_opt_current, n_opt_current, Fd_min_current, x_opt_current

def generate_energy(E_mean, E_std, N):
    """生成N个domain的特征：折叠长度"""
    E_samples = []
    
    # 持续采样直到获得N个符合条件的样本
    while len(E_samples) < N:
        # 从高斯分布中采样
        sample = np.random.normal(E_mean, E_std)
        
        # 检查样本是否在允许范围内
        if lower_bound <= sample <= upper_bound:
            E_samples.append(sample)
    
    return np.array(E_samples)

def main():
    # step 1: 生成domain特征参数
    print("生成domain特征参数...")
    DeltaE = generate_energy(E_mean, E_std, N)
    
    # step 2: 对每个domain进行自适应优化
    print(f"\n开始对 {N} 个domain进行自适应优化...")
    
    # 存储每个domain的结果
    all_results = []
    
    for i in range(N):
        # 优化单个domain
        f_vals_i, r_opt_i, n_opt_i, Fd_min_i, x_opt_i = optimize_single_domain_adaptive(DeltaE[i], i)
        
        # 存储结果
        domain_result = {
            'domain_idx': i,
            'xi_f': xi_f,
            'f_vals': f_vals_i,
            'r_opt': r_opt_i,
            'n_opt': n_opt_i,
            'Fd_min': Fd_min_i,
            'x_opt': x_opt_i,
            'DeltaE': DeltaE[i],
            'num_points': len(f_vals_i)
        }
        
        all_results.append(domain_result)
        
        # 保存单个domain的结果
        domain_df = pd.DataFrame({
            'f': f_vals_i,
            'r_opt': r_opt_i,
            'n_opt': n_opt_i,
            'Fd_min': Fd_min_i,
            'x_opt': x_opt_i
        })
        
        domain_df.to_csv(os.path.join(save_path, f"domain_{i+1}_results.csv"), index=False)
        print(f"  Domain {i+1} 结果已保存，点数: {len(f_vals_i)}\n")
    
    # step 3: 创建统一的力值网格（所有domain力值点的并集）
    print("创建统一的力值网格...")
    all_f_vals = []
    for result in all_results:
        all_f_vals.extend(result['f_vals'])
    
    unified_f_vals = np.unique(np.sort(all_f_vals))
    print(f"统一力值网格点数: {len(unified_f_vals)}")
    
    # step 4: 在统一力值网格上重新优化所有domain（为了比较和统一分析）
    print("在统一力值网格上重新优化所有domain...")
    unified_r_opt = np.zeros((len(unified_f_vals), N))
    unified_n_opt = np.zeros((len(unified_f_vals), N))
    unified_Fd_min = np.zeros((len(unified_f_vals), N))
    unified_x_opt = np.zeros((len(unified_f_vals), N))
    
    for i in range(N):
        print(f"  重新优化 domain {i+1}...")
        for j, f in enumerate(unified_f_vals):
            r_opt, n_opt, Fd_min, x_opt = optimize_single_point(f, DeltaE[i])
            unified_r_opt[j, i] = r_opt
            unified_n_opt[j, i] = n_opt
            unified_Fd_min[j, i] = Fd_min
            unified_x_opt[j, i] = x_opt
    
    # step 5: 保存统一网格结果
    print("保存统一网格结果...")
    df_r = pd.DataFrame(unified_r_opt, index=unified_f_vals, columns=[f"Domain_{i+1}" for i in range(N)])
    df_n = pd.DataFrame(unified_n_opt, index=unified_f_vals, columns=[f"Domain_{i+1}" for i in range(N)])
    df_Fd = pd.DataFrame(unified_Fd_min, index=unified_f_vals, columns=[f"Domain_{i+1}" for i in range(N)])
    df_x = pd.DataFrame(unified_x_opt, index=unified_f_vals, columns=[f"Domain_{i+1}" for i in range(N)])
    
    df_r.to_csv(os.path.join(save_path, "r_values_unified.csv"))
    df_n.to_csv(os.path.join(save_path, "n_values_unified.csv"))
    df_Fd.to_csv(os.path.join(save_path, "Fd_values_unified.csv"))
    df_x.to_csv(os.path.join(save_path, "x_values_unified.csv"))
    
    # 保存domain的特征参数
    df_energy = pd.DataFrame({
        'Domain': [f"Domain_{i+1}" for i in range(N)],
        'xi_f': xi_f,
        'DeltaE': [f"{DeltaE[i]:.2f}" for i in range(N)],
        'Num_Points': [result['num_points'] for result in all_results]
    })
    df_energy.to_csv(os.path.join(save_path, "domain_parameters.csv"), index=False)
    
    # 保存自适应细化参数
    params_df = pd.DataFrame({
        'Parameter': ['xi_f', 'alpha', 'E_mean', 'E_std', 'N', 'r_grid', 'n_grid',
                     'f_grid_initial', 'f_max', 'refinement_threshold', 
                     'max_refinement_level', 'tolerance', 'upper_bound', 'lower_bound'],
        'Value': [xi_f, alpha, E_mean, E_std, N, r_grid, n_grid,
                 f_grid_initial, f_max, refinement_threshold, 
                 max_refinement_level, tolerance, upper_bound, lower_bound]
    })
    params_df.to_csv(os.path.join(save_path, "adaptive_parameters.csv"), index=False)
    
    print(f"\n优化完成！")
    print(f"结果已保存到 '{save_path}' 目录")
    print(f"\n生成的文件:")
    print(f"  - domain_X_results.csv: 每个domain的自适应优化结果 (X=1-{N})")
    print(f"  - r_values_unified.csv: 统一网格下的最优端到端距离")
    print(f"  - n_values_unified.csv: 统一网格下的最优展开分数")
    print(f"  - Fd_values_unified.csv: 统一网格下的最小自由能")
    print(f"  - x_values_unified.csv: 统一网格下的最优端到端因子")
    print(f"  - domain_parameters.csv: domain的特征参数和优化统计")
    print(f"  - adaptive_parameters.csv: 自适应细化参数")
    
    # 返回结果供后续使用
    return {
        'xi_f': xi_f,
        'unified_f_vals': unified_f_vals,
        'unified_r_opt': unified_r_opt,
        'unified_n_opt': unified_n_opt,
        'unified_Fd_min': unified_Fd_min,
        'unified_x_opt': unified_x_opt,
        'all_results': all_results
    }

# 主程序入口
if __name__ == "__main__":
    results = main()