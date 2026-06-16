'''
单个domain的模拟和理论对比 Delta理论
科研风格可视化
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import fsolve

# ============ 字体路径 ============
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
    # 将字体文件添加到matplotlib的字体管理器中
    fm.fontManager.addfont(font_path)
    # 获取字体的属性
    font_prop = fm.FontProperties(fname=font_path)
    # 将字体的名称设置为默认字体
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

# ============ 参数设置 ============
xi_f = 5.0  # 折叠态轮廓长度
alpha = 7.0  # 
DeltaE = 13.0  # 能量差 ΔE
DeltaEt = 19.3
rs = 9.060530  # 反应距离
pi = np.pi

# ============ 辅助函数定义 ============
def pf_from_force(f):
    """根据力f计算折叠概率p_f"""
    # p_f(f) = 1 / (1 + exp(-ΔE + f·r_s))
    exponent = -DeltaEt + f * rs
    if exponent > 100:  # 避免溢出
        return 0.0
    elif exponent < -100:  # 避免下溢
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(exponent))

def force_from_r_and_Lc(r, Lc):
    """根据r和Lc计算力f"""
    # x = r / L_c(f)
    x = r / Lc
    
    # 避免数值问题
    if x >= 0.999:
        x = 0.999
    elif x <= 0:
        x = 0.001
    
    # f = -π²x/L_c² + 4x/[π(1-x²)²]
    term1 = -(pi**2 * x) / (Lc**2)
    term2 = (4 * x) / (pi * (1 - x**2)**2)
    return term1 + term2

def self_consistent_equations(f, r):
    """自洽方程组：定义需要满足的方程
    
    根据图片中的方程：
    1. p_f = 1/(1+exp(-ΔE + f·r_s))
    2. L_c = ξ_u - p_f(ξ_u - ξ_f)
    3. x = r/L_c
    4. f_calculated = -π²x/L_c² + 4x/[π(1-x²)²]
    
    我们需要 f = f_calculated
    所以方程是：f - f_calculated = 0
    """
    # 计算p_f
    pf = pf_from_force(f)
    
    # 计算L_c
    xi_f + (1 - pf)*(alpha - 1)*xi_f
    
    # 检查Lc是否有效
    if Lc <= 0:
        # 返回一个大的残差，表示不可行
        return 1e6
    
    # 计算x
    x = r / Lc
    
    # 避免数值问题
    if x >= 0.999:
        x = 0.999
    elif x <= 0:
        x = 0.001
    
    # 计算基于当前Lc和x的力
    f_calc = -(pi**2*x) / (Lc**2) + (4*x) / (pi*(1 - x**2)**2)
    
    # 返回残差：f - f_calc
    return f - f_calc

def solve_self_consistent(r, f_guess):
    """使用fsolve求解自洽方程，使用给定的初始猜测"""
    try:
        # 使用fsolve求解，使用给定的初始猜测
        f_solution = fsolve(self_consistent_equations, f_guess, args=(r,))
        
        # 检查解是否有效
        if f_solution[0] < 0:
            # 如果力为负，尝试从0开始
            f_solution = fsolve(self_consistent_equations, 0.0, args=(r,))
        
        # 获取最终解
        f_final = f_solution[0]
        
        # 计算对应的p_f和n_u
        pf = pf_from_force(f_final)
        n_u = 1.0 - pf
        
        return n_u, f_final
    except Exception as e:
        print(f"在r={r}处求解失败: {e}")
        # 返回边界值
        if r < xi_f:
            return 0.0, 0.0
        else:
            return 1.0, 0.0

# ============ 模拟部分（基于均匀扫描） ============

def Lc(n):
    """轮廓长度作为n的函数"""
    return xi_f + n*(alpha - 1)*xi_f

def x(r, n):
    """端到端距离与轮廓长度之比"""
    return r / Lc(n)

def F_WLC(r, n):
    """WLC自由能"""
    Lc_val = Lc(n)
    x_val = x(r, n)
    if x_val >= 1.0 or x_val < 0:  # 避免数值问题
        return np.inf
    term1 = (pi**2) / (2 * Lc_val) * (1 - x_val**2)
    term2 = (2 * Lc_val) / (pi * (1 - x_val**2))
    return term1 + term2

def F_MS(r, n):
    """WLC自由能"""
    Lc_val = Lc(n)
    x_val = x(r, n)
    x_vals = np.clip(x_val, 0.001, 0.999)
    return 0.25*Lc_val*x_vals**2*(3 - 2*x_vals)/(1 - x_vals)


def force_MS(r,n):
    Lc_val = Lc(n)
    x_val = x(r, n)
    x_vals = np.clip(x_val, 0.001, 0.999)
    return 0.25*((1 - x_vals)**(-2) + 4*x_vals - 1.0)

def U(n):
    """周期性势能项"""
    return DeltaE * n - DeltaE * np.cos(2 * pi * n)

def F_total(r, n):
    """总自由能"""
    return F_MS(r, n) + U(n)

def f_WLC(r, n):
    """WLC力"""
    Lc_val = Lc(n)
    x_val = x(r, n)
    if x_val >= 1.0 or x_val <= 0:  # 避免数值问题
        return np.inf
    term1 = - (pi**2 * x_val) / (Lc_val**2)
    term2 = (4 * x_val) / (pi * (1 - x_val**2)**2)
    return term1 + term2

def simulate_n_f_scan(r_values, n_points=1000):
    """对于每个r，通过均匀扫描n得到模拟结果"""
    n_sim = []
    f_sim = []
    
    # 创建n的均匀网格
    n_grid = np.linspace(0, 1, n_points)
    
    for r in r_values:
        # 计算所有n对应的自由能
        F_values = np.zeros_like(n_grid)
        valid_indices = []
        
        for i, n in enumerate(n_grid):
            F_val = F_total(r, n)
            if np.isfinite(F_val):  # 只考虑有效的自由能值
                F_values[i] = F_val
                valid_indices.append(i)
            else:
                F_values[i] = np.inf
        
        # 如果所有值都无效，使用边界值
        if len(valid_indices) == 0:
            n_opt = 0.0 if F_total(r, 0) < F_total(r, 1) else 1.0
        else:
            # 找到最小自由能对应的n
            min_idx = valid_indices[np.argmin(F_values[valid_indices])]
            n_opt = n_grid[min_idx]
        
        # 计算对应的力
        f_val = F_MS(r, n_opt)
        
        n_sim.append(n_opt)
        f_sim.append(f_val)
    
    return np.array(n_sim), np.array(f_sim)

# ============ 理论部分（基于图片中的自洽方程，使用fsolve求解） ============
def theoretical_n_f_fsolve(r_values):
    """基于图片中自洽方程的理论解，使用fsolve求解
    修改：将上一次迭代的结果作为新的猜测解进行迭代"""
    n_theory = []
    f_theory = []
    
    # 对r值进行排序，确保是单调递增的
    r_sorted = np.sort(r_values)
    
    # 第一个点的初始猜测
    if r_sorted[0] < 5:
        f_guess = 0.1
    elif r_sorted[0] < 10:
        f_guess = 1.0
    elif r_sorted[0] < 15:
        f_guess = 5.0
    else:
        f_guess = 10.0
    
    for i, r in enumerate(r_sorted):
        # 使用fsolve求解自洽方程，使用当前的f_guess作为初始猜测
        n_u, f_val = solve_self_consistent(r, f_guess)
        n_theory.append(n_u)
        f_theory.append(f_val)
        
        # 将当前解作为下一次迭代的初始猜测
        f_guess = f_val
        
        # 显示进度
        if (i+1) % 50 == 0 or i == 0 or i == len(r_sorted)-1:
            print(f"  进度: {i+1}/{len(r_sorted)}, r={r:.2f}, n_u={n_u:.4f}, f={f_val:.4f}, 猜测={f_guess:.4f}")
    
    # 由于我们可能对r进行了排序，需要将结果按照原始r_values的顺序返回
    # 这里我们假设r_values已经是单调递增的，所以不需要重新排序
    return np.array(n_theory), np.array(f_theory)

# ============ 主程序 ============
def main():
    # 创建r值数组
    r_min = 0.0
    r_max = 0.95*alpha*xi_f  # 避免r=0和r>=Lc的边界问题
    r_values = np.linspace(r_min, r_max, 500)  # 减少点数以加快计算
    
    # 计算模拟结果（使用均匀扫描）
    print("开始模拟计算...")
    n_sim, f_sim = simulate_n_f_scan(r_values, n_points=2000)
    print("模拟计算完成")
    
    # 计算理论结果（基于图片中的自洽方程，使用fsolve）
    print("开始理论计算（使用fsolve，连续迭代）...")
    # n_theory, f_theory = theoretical_n_f_fsolve(r_values)
    print("理论计算完成")
    
    # ============ 科研风格可视化 ============
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. n-r 曲线
    ax1 = axes[0]
    # 模拟曲线 - 使用变量中定义的线宽
    ax1.plot(r_values, n_sim, color='red', linewidth=lines_linewidth, label='Optimization')
    # 理论曲线 - 使用变量中定义的线宽
    # ax1.plot(r_values, n_theory, color='blue', linewidth=lines_linewidth, linestyle='--', label='Theory')
    
    # 使用变量中定义的字体大小和样式
    ax1.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax1.set_ylabel('Unfolding probability $n$', fontsize=label_fontsize)
    ax1.set_title('$n$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    
    # 设置网格 - 使用变量中定义的线宽和透明度
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例 - 使用变量中定义的字体大小
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax1.set_xlim(0, 19)
    ax1.set_ylim(-0.1, 1.1)
    
    # 设置刻度（所有边框都有刻度，朝内） - 使用变量中定义的参数
    ax1.tick_params(axis='both', which='major', 
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width,       # 刻度线宽
                   length=xtick_major_size)       # 刻度线长度
    
    ax1.tick_params(axis='both', which='minor',
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width*0.75,  # 次刻度线宽稍小
                   length=xtick_major_size*0.5)   # 次刻度线长稍小
    
    # 开启次刻度
    ax1.minorticks_on()
    
    # 2. f-r 曲线
    ax2 = axes[1]
    # 模拟曲线 - 使用变量中定义的线宽
    ax2.plot(r_values, f_sim, color='red', linewidth=lines_linewidth, label='Optimization')
    # 理论曲线 - 自洽方程解（fsolve）
#    ax2.plot(r_values, f_theory, color='blue', linewidth=lines_linewidth, linestyle='--', label='Theory')
    
    # 使用变量中定义的字体大小和样式
    ax2.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax2.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax2.set_title('Force $f$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    
    # 设置网格 - 使用变量中定义的线宽和透明度
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例 - 使用变量中定义的字体大小
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-1.0, 10)
    
    # 设置刻度（所有边框都有刻度，朝内） - 使用变量中定义的参数
    ax2.tick_params(axis='both', which='major', 
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width,       # 刻度线宽
                   length=xtick_major_size)       # 刻度线长度
    
    ax2.tick_params(axis='both', which='minor',
                   direction=xtick_direction,     # 刻度朝内
                   top=xtick_top,                 # 顶部显示刻度
                   right=ytick_right,             # 右侧显示刻度
                   bottom=True,                   # 底部显示刻度
                   left=True,                     # 左侧显示刻度
                   width=xtick_major_width*0.75,  # 次刻度线宽稍小
                   length=xtick_major_size*0.5)   # 次刻度线长稍小
    
    # 开启次刻度
    ax2.minorticks_on()
    
    # 为两个子图添加边框强化 - 使用变量中定义的线宽
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)
    
    # 调整子图间距
    plt.subplots_adjust(wspace=0.25, top=0.9, bottom=0.1)
    
    plt.tight_layout()
    
    # 保存图形（多种格式）
    base_save_path = '/home/tyt/project/Single-chain/opt+R/Single_domain/simulation_results/theory_vs_numerical'
    
    # 保存为PNG（用于文档）- 使用变量中定义的分辨率
    plt.savefig(f'{base_save_path}.png', dpi=savefig_dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n图形已保存至: {base_save_path}.png")
    
    # ============ 输出数值结果示例 ============
    print("\n" + "="*80)
    print("参数设置:")
    print(f"  ξ_f = {xi_f}")
    print(f"  ΔE = {DeltaE}")
    print(f"  r_s = {rs}")
    print(f"\nr范围: {r_min:.2f} 到 {r_max:.2f}")
    print("\n自洽方程求解参数 (fsolve):")
    print(f"  r点数: {len(r_values)}")
    print(f"  初始猜测策略: 连续性假设，使用上一次的解作为下一次的猜测")



# ============ 运行主程序 ============
if __name__ == "__main__":
    main()