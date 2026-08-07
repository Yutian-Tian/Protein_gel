"""
单独分析N=2的情况，绘制应力-应变曲线和展开分数-应变曲线
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys

# ============ 字体设置 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
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
xi_f = 3.6          # 折叠态持续长度
alpha = 7.6         # 解折叠系数

E_mean = 11.9       # 平均能量差
E_std = 1.7         # 能量差的标准差

N_val = 2.0       # domain 的数量
N_test = [2.0,3.0]
M = 300
k1 = 6.5
k2 = 1.48
kR = 2.68           # 初始首末端距离 R0 = kR * N**0.5
lambda_max = 30.0  # 最大伸长比
Stress_max = 40.0  # 最大应力值

def MSforce(x):
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def R0ms(N):
    L = N*xi_f
    Rms = np.sqrt(2*L*(1 - 1/L*(1 - np.exp(-L))))

    return Rms

def load_average_curve_data(file_path):
    """
    从指定的CSV文件中加载平均曲线数据。
    :param file_path: CSV文件路径
    :return: f_val, r_val 两个numpy数组，分别表示第一列f和第二列r的值
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径。")

    data = pd.read_csv(file_path)
    if len(data.columns) < 2:
        raise ValueError("CSV文件必须至少包含两列数据。")
    f_val = data.iloc[:, 0].values
    r_val = data.iloc[:, 1].values
    n_val = data.iloc[:, 2].values

    return f_val, r_val, n_val

def n_theory(f):
    n = 0.5 + 0.5*np.tanh(k1*(f - k2))
    return n

def Plot_n(N_val, N_test, save_dir=None):

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    filepath = f"/home/tyt/project/protein_gel/GB1_results/Multi_chains/N_{int(N_val)}_M_{M}_test_results/average_curves.csv"
    f_val, r_val, n_val = load_average_curve_data(filepath)
    ax.plot(f_val, n_val/N_val, 'o', color='blue', markerfacecolor='none',
            markeredgewidth=2, markersize=10,
            label=f'N={int(N_val)}', zorder=4)
    n_theo = n_theory(f_val)
    ax.plot(f_val, n_theo, '-', color='black', linewidth=lines_linewidth, label=f'Theory', zorder=4)

    # 初始位置 未解折叠的初始相对拉伸
    # 为不同 N 分配颜色
    colors = ['#FF0000', '#FFD700', '#00FF00', '#1E90FF']
    for idx, n in enumerate(N_test):
        # R0 = kR * N**0.5
        R0 = R0ms(n)
        x_init = R0/(n*xi_f)
        f_init = MSforce(x_init)
        n_init = n_theory(f_init)
        ax.plot(f_init, n_init, 's', color=colors[idx], markerfacecolor='none',
                markeredgewidth=4, markersize=15,
                label=f'$x_0$: N={int(n)}', zorder=4)
        
        # 标签与标题
    ax.set_xlabel('force $f$', fontsize=label_fontsize)
    ax.set_ylabel('Unfolding fraction $n/N$', fontsize=label_fontsize)
    ax.set_title(f'Unfolding fraction vs. strain', 
                  fontsize=title_fontsize, pad=20)

    # 网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例：模拟点和理论线合并为一个图例项（通过句柄去重实现）
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-0.1, 1.1)

    ax.tick_params(axis='both', which='major',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5,
                    labelbottom=False, labelleft=False)
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'N={int(N_val)}_n_compare.png')
        fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"展开分数曲线已保存至: {save_path}") 


def main():
    print("=" * 80)
    print("开始计算...")
    print("=" * 80)

    data_dir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_N0.5"  # 可修改为你希望的输出路径
    output_dir = data_dir
    Plot_n(N_val, N_test, save_dir=output_dir)

    print("=" * 80)
    print("计算完成。")
    print("=" * 80)

if __name__ == "__main__":
    main()