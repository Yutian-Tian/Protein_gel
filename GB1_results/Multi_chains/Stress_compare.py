"""
改变domain的数量，比较本构曲线
改变R0的值，比较本构曲线
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
lines_markersize = 35

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
xi_f = 3.6  # 折叠态持续长度
alpha = 7.6      # 解折叠系数

E_mean = 11.9  # 平均能量差
E_std = 1.7    # 能量差的标准差

N_val = [10.0]     # domain 的数量
M = 300
k1 = 6.5
k2 = 1.50
R0 = 5.0    # 初始首末端距离
lambda_max = 40.0  # 最大伸长比


def load_average_curve_data(file_path):
    """
    从指定的CSV文件中加载平均曲线数据。
    :param file_path: CSV文件路径
    :return: f_val, r_val 两个numpy数组，分别表示第一列f和第二列r的值
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径。")

    # 读取CSV，默认第一行为标题，从第二行开始读取数据
    data = pd.read_csv(file_path)

    # 确保至少有两列
    if len(data.columns) < 2:
        raise ValueError("CSV文件必须至少包含两列数据。")

    # 第一列 → f_val，第二列 → r_val
    f_val = data.iloc[:, 0].values
    r_val = data.iloc[:, 1].values

    return f_val, r_val

def StressOptimization(R0, N, r_val, f_val):
    Re = R0
    r_val = np.asarray(r_val)
    f_val = np.asarray(f_val)

    # 只保留 r >= R0 的数据点
    mask = r_val >= Re
    if not np.any(mask):
        raise ValueError("没有找到 r >= R0 的数据点，请检查 R0 或数据范围。")

    r_selected = r_val[mask]
    lambda_ = r_selected / Re                    # λ = r / R0
    r2 = lambda_ ** (-0.5) * Re                  # 对应的另一条链伸长

    # 线性插值获取力值，超出范围时使用边界值
    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])

    # 可选：检查外推警告
    if np.any(r_selected < r_val[0]) or np.any(r_selected > r_val[-1]):
        print("警告：某些 r1 值超出原始数据范围，使用了边界值。")
    if np.any(r2 < r_val[0]) or np.any(r2 > r_val[-1]):
        print("警告：某些 r2 值超出原始数据范围，使用了边界值。")

    # 计算应力：σ = Re [ F'(λRe) - λ^{-3/2} F'(λ^{-1/2}Re) ]
    sigma = Re * (f1 - lambda_ ** (-1.5) * f2)

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

def create_visualization(save_dir=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    for N in N_val:
        filepath = f"/home/tyt/project/protein_gel/GB1_results/Multi_chains/N_{int(N)}_M_{M}_results/average_curves.csv"
        f_val, r_val = load_average_curve_data(filepath)
        lambda_, sigma = StressOptimization(R0, N, r_val, f_val)
        ax.scatter(lambda_, sigma, label=f'N={int(N)}', s=lines_markersize, alpha=0.8)

    # 设置标签和标题
    ax.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Stress $\sigma/\\rho k_B T$', fontsize=label_fontsize)
    ax.set_title(f'Constitutive curve: $R_0={R0:.0f}$', 
                  fontsize=title_fontsize, pad=20)
    # 设置网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    # ax.set_xlim(1.0, lambda_max)
    ax.set_xlim(30.0, 38.0)
    ax.set_ylim(8.0, 12.0)
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax.minorticks_on()
    
    # 强化边框
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    # 保存第三幅图
    if save_dir:
        save_path3 = os.path.join(save_dir, 'Stress_compare.png')
        fig.savefig(save_path3, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"本构曲线已保存至: {save_path3}")


def main():
    print("=" * 80)
    print("开始生成本构曲线比较图...")
    print("=" * 80)
    
    # ============ 在这里指定文件路径 ============
    data_dir = "/home/tyt/project/protein_gel/GB1_results/Multi_chains"
    output_dir = data_dir  # 保存结果的目录
    
    # 创建可视化并保存
    create_visualization(save_dir=output_dir)
    
    print("=" * 80)
    print("本构曲线比较图生成完成。")
    print("=" * 80)

if __name__ == "__main__":
    main()