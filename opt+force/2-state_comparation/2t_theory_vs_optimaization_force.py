'''
单个domain的模拟和理论对比 Delta理论
修改要求：
1. 去掉求解力f的自洽方程步骤
2. 从自由能最小化找到跃变点
3. 计算DeltaEt和反应距离rs
4. 可视化p_u-f曲线和n_u/N-f曲线
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy import interpolate

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

# ============ 参数设置 ============
xi_f = 5.0  # 折叠态轮廓长度
alpha = 7.0 # 解折叠比 
DeltaE = 13.0  # 能量差 ΔE
pi = np.pi
f_min = 0.0
f_max = 6.0
f_grid = 200
r_grid = 400
n_grid = 200

# ============ 辅助函数定义 ============
def Lc(n):
    """轮廓长度作为n的函数"""
    return xi_f + n * (alpha - 1)*xi_f

def x(r, n):
    """端到端距离与轮廓长度之比"""
    return r / Lc(n)

def force_WLC(r, n):
    """根据r和Lc计算力f,支持numpy数组输入"""
    # 计算轮廓长度
    Lc_vals = Lc(n)
    
    # 计算x = r / Lc，并进行数值保护
    # 使用clip将x限制在(0.001, 0.999)范围内
    x_vals = np.clip(r / Lc_vals, 0.001, 0.999)
    
    # 计算力f
    term1 = -(pi**2 * x_vals) / (Lc_vals**2)
    term2 = (4 * x_vals) / (pi * (1 - x_vals**2)**2)
    f = term1 + term2
    
    # 可选：对接近边界的值进行特殊处理
    # 如果x接近0.999，力应该很大
    near_boundary = (r / Lc_vals) > 0.995
    f[near_boundary] = 1000.0  # 或者根据模型设定一个合适的值
    
    return f

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

def F_modified(r, n, f):
    """修正自由能：F(r,n) - f·r"""
    return F_total(r, n) - f * r

def pf_theory(f, DeltaEt, rs):
    """理论折叠概率"""
    exponent = -DeltaEt + f * rs
    if exponent > 50:  # 避免溢出
        return 0.0
    elif exponent < -50:  # 避免下溢
        return 1.0
    else:
        return 1.0 / (1.0 + np.exp(exponent))

def pu_theory(f, DeltaEt, rs):
    """理论展开概率"""
    return 1.0 - pf_theory(f, DeltaEt, rs)

# ============ 模拟部分（基于修正自由能最小化） ============
def simulate_minimize_F_modified(f_values, r_points=200, n_points=100):
    """对于每个给定的力f，通过扫描r和n最小化修正自由能F(r,n;f) = F_total(r,n) - f·r"""
    n_sim = []
    r_sim = []
    
    # 创建r和n的网格
    r_min = 0.0
    r_max = 0.95*alpha*xi_f  # r的最大值
    r_vals = np.linspace(r_min, r_max, r_points)
    n_vals = np.linspace(0, 1.0, n_points)
    
    for f in f_values:
        min_F = np.inf
        r_opt = 0.0
        n_opt = 0.0
        
        # 扫描r和n
        for r in r_vals:
            for n in n_vals:
                # 计算修正自由能
                F_mod = F_modified(r, n, f)
                
                # 检查是否有效且更小
                if np.isfinite(F_mod) and F_mod < min_F:
                    min_F = F_mod
                    r_opt = r
                    n_opt = n
        
        r_sim.append(r_opt)
        n_sim.append(n_opt)
    
    return np.array(r_sim), np.array(n_sim)

def find_jump_points(r_sim, f_sim, n_sim):
    """找到r跃变点和对应的力f"""
    # 计算r的变化率
    r_diff = np.abs(np.diff(r_sim))
    f_diff = np.diff(f_sim)
    
    # 找到r变化最大的点（跃变点）
    # 使用阈值方法：r变化超过平均值3个标准差
    mean_r_diff = np.mean(r_diff)
    std_r_diff = np.std(r_diff)
    threshold = mean_r_diff + 3 * std_r_diff
    
    # 找到所有超过阈值的点
    jump_indices = np.where(r_diff > threshold)[0]
    
    if len(jump_indices) > 0:
        # 取第一个跃变点
        jump_idx = jump_indices[0]
        
        # 跃变前的点（索引减1）
        r_front = r_sim[jump_idx]
        f_jump = f_sim[jump_idx]
        n_front = n_sim[jump_idx]
        
        # 跃变后的点（索引加1）
        if jump_idx + 1 < len(r_sim):
            r_behind = r_sim[jump_idx + 1]
            n_behind = n_sim[jump_idx + 1]
        else:
            # 如果跃变点是最后一个点，使用下一个f值的模拟结果
            # 这需要重新计算，但这里简单处理
            r_behind = r_sim[jump_idx] + 0.1
            n_behind = n_sim[jump_idx]
        
        print(f"发现跃变点: f={f_jump:.4f}")
        print(f"  跃变前: r={r_front:.4f}, n={n_front:.4f}")
        print(f"  跃变后: r={r_behind:.4f}, n={n_behind:.4f}")
        
        return f_jump, r_front, n_front, r_behind, n_behind
    else:
        print("未发现明显的跃变点，使用最大r值作为跃变点")
        # 如果没有明显的跃变点，使用最后一个点
        last_idx = len(f_sim) - 1
        f_jump = f_sim[last_idx]
        r_front = r_sim[last_idx-1]
        n_front = n_sim[last_idx-1]
        r_behind = r_sim[last_idx]
        n_behind = n_sim[last_idx]
        
        return f_jump, r_front, n_front, r_behind, n_behind

# ============ 主程序 ============
def main():
    # 创建f值数组（模拟部分的输入）
    f_values = np.linspace(f_min, f_max, f_grid)
    
    # 计算模拟结果（基于修正自由能最小化）
    print("开始模拟计算（修正自由能最小化）...")
    r_sim, n_sim = simulate_minimize_F_modified(f_values, r_points=r_grid, n_points=n_grid)
    print("模拟计算完成")
    
    # 找到跃变点
    print("\n寻找跃变点...")
    f_jump, r_front, n_front, r_behind, n_behind = find_jump_points(r_sim, f_values, n_sim)
    
    # 计算DeltaEt和反应距离rs
    print("\n计算理论参数...")
    F00 = F_total(0.0, 0.0)
    F_behind = F_total(r_behind, n_behind)
    F_front = F_total(r_front, n_front)
    DeltaEt1 = F_behind - F00
    DeltaEt2 = F_behind - F_front
    rs1 = r_behind
    rs2 = r_behind - r_front
    
    
    # 计算理论p_u(f)曲线
    print("\n计算理论p_u(f)曲线...")
    pu_theory_vals1 = np.array([pu_theory(f, DeltaEt1, rs1) for f in f_values])
    pu_theory_vals2 = np.array([pu_theory(f, DeltaEt2, rs2) for f in f_values])

    f_theory_vals = force_MS(r_sim, pu_theory_vals1)
    
    # 准备模拟的p_u(f)曲线
    pu_sim = n_sim   # n_u/N
    
    # ============ 科研风格可视化 ============
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. p_u-f 和 n_u/N-f 曲线
    ax1 = axes[0]
    # 模拟的n_u/N-f曲线
    ax1.plot(f_values, pu_sim, color='red', linewidth=lines_linewidth, label=r'Optimization')
    
    # 标记跃变点
    if f_jump is not None:
        # 找到最接近跃变力的索引
        jump_idx = np.argmin(np.abs(f_values - f_jump))
        ax1.scatter(f_values[jump_idx], pu_sim[jump_idx], color='purple', 
                   s=200, zorder=5, label=f'$f_t$={f_jump:.2f}')
        
    ax1.plot(f_values, pu_theory_vals1, color='blue', linewidth=lines_linewidth, linestyle='--', label=f'$\Delta E_t$={DeltaEt1:.1f},$r_s$ = {rs1:.1f}')
    ax1.plot(f_values, pu_theory_vals2, color='purple', linewidth=lines_linewidth, linestyle='--', label=f'$\Delta E_t$={DeltaEt2:.1f},$r_s$ = {rs2:.1f}')
    
    ax1.set_xlabel('Force $f$', fontsize=label_fontsize)
    ax1.set_ylabel('Unfolding probability $n$', fontsize=label_fontsize)
    ax1.set_title('Unfolding probability $n$ vs. force $f$', fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax1.set_xlim(f_min, f_max)
    ax1.set_ylim(-0.1, 1.1)
    
    # 设置刻度
    ax1.tick_params(axis='both', which='major', 
                   direction=xtick_direction,
                   top=xtick_top,
                   right=ytick_right,
                   width=xtick_major_width,
                   length=xtick_major_size)
    
    # 2. f-r 曲线（模拟结果）
    ax2 = axes[1]
    ax2.plot(r_sim, f_values, color='red', linewidth=lines_linewidth, label='Optimization')
    # ax2.plot(r_sim, f_theory_vals, color='blue', linewidth=lines_linewidth,  linestyle='--',label='Theory')
    
    # 标记跃变点
    if f_jump is not None:
        # 找到跃变点对应的r值（前后两个点）
        jump_idx = np.argmin(np.abs(f_values - f_jump))
        ax2.scatter(r_sim[jump_idx], f_values[jump_idx], color='purple', s=200, 
                   zorder=5, label=f'$r_{{front}}$={r_front:.2f}')
        ax2.scatter(r_behind, f_jump, color='green', s=200, 
                   zorder=5, label=f'$r_{{behind}}$={r_behind:.2f}')
    
    ax2.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax2.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax2.set_title('Force $f$ vs. distance $r$', fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax2.set_xlim(0, alpha*xi_f)
    ax2.set_ylim(-0.5, f_max)
    
    # 设置刻度
    ax2.tick_params(axis='both', which='major', 
                   direction=xtick_direction,
                   top=xtick_top,
                   right=ytick_right,
                   width=xtick_major_width,
                   length=xtick_major_size)
    
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


    # 为两个子图添加边框强化
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)
    
    # 调整子图间距
    plt.subplots_adjust(wspace=0.25, top=0.9, bottom=0.1)
    
    plt.tight_layout()
    
    # 保存图形
    base_save_path = '/home/tyt/project/Single-chain/opt+force/2-state_comparation/results/theory_vs_numerical_modified'
    
    # 保存为PNG
    plt.savefig(f'{base_save_path}.png', dpi=savefig_dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n图形已保存至: {base_save_path}.png")
    
    # ============ 输出数值结果示例 ============
    print("\n" + "="*80)
    print("参数设置:")
    print(f"  ξ_f = {xi_f}")
    print(f"  ΔE = {DeltaE}")
    print(f"\n跃变点信息:")
    print(f"  跃变力 f_jump = {f_jump:.4f}")
    print(f"  跃变前: r_front = {r_front:.4f}, n_front = {n_front:.4f}, n_front/N = {n_front:.4f}")
    print(f"  跃变后: r_behind = {r_behind:.4f}, n_behind = {n_behind:.4f}, n_behind/N = {n_behind:.4f}")
    print(f"\n理论参数:")
    print(f"  F(0,0) = {F00:.4f}")
    print(f"  F(r_behind, n_behind) = {F_behind:.4f}")
    print(f"  DeltaEt1 = {DeltaEt1:.4f}")
    print(f"  DeltaEt2 = {DeltaEt2:.4f}")
    print(f"  反应距离 rs1 = {rs1:.4f}")
    print(f"  反应距离 rs2 = {rs2:.4f}")
    print(f"\n模拟参数:")
    print(f"  f点数: {len(f_values)}")
    print(f"  r扫描点数: {r_grid}")
    print(f"  n扫描点数: {n_grid}")

# ============ 运行主程序 ============
if __name__ == "__main__":
    main()