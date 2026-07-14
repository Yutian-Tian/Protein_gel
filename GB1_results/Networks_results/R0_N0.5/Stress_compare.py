"""
改变domain的数量，比较本构曲线
改变R0的值，比较本构曲线
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

N_val = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]   # domain 的数量
M = 300
k1 = 6.5
k2 = 1.48
kR = 2.68           # 初始首末端距离 R0 = kR * N**0.5
lambda_max = 30.0  # 最大伸长比
Stress_max = 40.0  # 最大应力值

def Lc(f, N):
    """
    物理含义为：施加外力f时的平均轮廓长度
    """
    contour_length = N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))
    return contour_length

def MSforce(x):
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def x_theory(lam, N):
    """
    新增函数：计算图1中的 x(lambda) 解析解
    根据图1的解析公式计算相对伸长率 x(lambda)。
    公式推导：x(λ) = (-B + sqrt(B^2 + 4*A*x0*λ)) / (2*A)
    参数说明：
        lam: 拉伸比 λ (可以是标量或 numpy 数组)
        N  : 链段数量 / domain 数量
    """
    # 1. 使用牛顿迭代法求解 xc，满足 fhat(xc) = k2 = 1.48
    xc = 0.5  # 初始猜测值
    for _ in range(100):
        f_val = 1/(4*(1-xc)**2) - 1/4 + xc - k2
        f_prime_val = 1/(2*(1-xc)**3) + 1
        xc_new = xc - f_val / f_prime_val
        if abs(xc_new - xc) < 1e-12:
            xc = xc_new
            break
        xc = xc_new

    # 2. 计算在 xc 处的导数 f'(xc)
    f_prime_xc = 1/(2*(1-xc)**3) + 1
    
    # 3. 计算中间参数 A, B, x0
    # 图1中的 μ，物理上对应代码中的 alpha（最大展开长度 / 折叠长度）
    A = (alpha - 1) / 2 * k1 * f_prime_xc
    B = (alpha + 1) / 2 - A * xc
    
    # 根据代码，R0 = kR * N**0.5，得出 x0 = R0 / (N * xi_f)
    x0 = kR / (np.sqrt(N) * xi_f) 
    
    # 4. 套用一元二次方程求根公式计算 x(lambda)
    inside_sqrt = B**2 + 4 * A * x0 * lam
    x_lambda_val = (-B + np.sqrt(inside_sqrt)) / (2 * A)
    
    return x_lambda_val

def PlotStressMS(R0, N=None):
    """
    使用Marko-Siggia模型绘制理论曲线
    若指定 N，则使用与 domain 数量相关的轮廓长度变化
    """
    x_MS = np.linspace(0.0, 0.99, 1000)
    f_MS = MSforce(x_MS)
    # 如果给了 N，就用对应的 Lc；否则采用固定轮廓长度（无去折叠参考线）
    if N is not None:
        r_MS = x_MS * Lc(f_MS, N)
    else:
        # 无去折叠参考线：轮廓长度固定为 N=1 完全展开时的值，这里取 alpha*xi_f
        L0 = xi_f * alpha
        r_MS = x_MS * L0
        # 标记一下，后面不参与 legend 中的 N 标签
    lambda_MS, sigma_MS = StressOptimization(R0, r_MS, f_MS)
    return lambda_MS, sigma_MS

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

def StressOptimization(R0, r_val, f_val):
    Re = R0
    r_val = np.asarray(r_val)
    f_val = np.asarray(f_val)

    mask = r_val >= Re
    if not np.any(mask):
        raise ValueError("没有找到 r >= R0 的数据点，请检查 R0 或数据范围。")

    r_selected = r_val[mask]
    lambda_ = r_selected / Re
    r2 = lambda_ ** (-0.5) * Re

    f1 = np.interp(r_selected, r_val, f_val, left=f_val[0], right=f_val[-1])
    f2 = np.interp(r2, r_val, f_val, left=f_val[0], right=f_val[-1])

    sigma = Re * (f1 - lambda_ ** (-1.5) * f2)

    eps = 1e-12
    if len(lambda_) == 0:
        return lambda_, sigma
    if np.abs(lambda_[0] - 1.0) > eps:
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        sigma[0] = 0.0
    return lambda_, sigma

def create_visualization(save_dir=None):
    # 颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(N_val)))

    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    for idx, N in enumerate(N_val):
        R0 = kR * N**0.5
        filepath = f"/home/tyt/project/protein_gel/GB1_results/Multi_chains/N_{int(N)}_M_{M}_test_results/average_curves.csv"
        f_val, r_val, n_val = load_average_curve_data(filepath)
        lambda_sim, sigma_sim = StressOptimization(R0, r_val, f_val)

        # 绘制模拟数据点：空心圆
        ax1.plot(lambda_sim, sigma_sim, 'o',
                color=colors[idx], markerfacecolor='none',
                markeredgewidth=2, markersize=8,
                label=f'N={int(N)}', zorder=4)

        # 绘制对应的理论曲线（实线）
        lam_th, sig_th = PlotStressMS(R0, N)
        ax1.plot(lam_th, sig_th, '-', color="black", linewidth=lines_linewidth, alpha=0.8, zorder=5)


    # 标签与标题
    ax1.set_xlabel('Stretch ratio $\\lambda$', fontsize=label_fontsize)
    ax1.set_ylabel('Stress $\\sigma/\\rho k_B T$', fontsize=label_fontsize)
    ax1.set_title(f'Constitutive curve: $R_0={kR:.2f} \sqrt{{N}}$', 
                  fontsize=title_fontsize, pad=20)

    # 网格
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例：模拟点和理论线合并为一个图例项（通过句柄去重实现）
    handles, labels = ax1.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax1.legend(by_label.values(), by_label.keys(),
              fontsize=legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    ax1.set_xlim(1.0, lambda_max)
    ax1.set_ylim(0.1, Stress_max)

    ax1.tick_params(axis='both', which='major',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    ax1.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5,
                    labelbottom=False, labelleft=False)
    ax1.minorticks_on()

    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        save_path1 = os.path.join(save_dir, f'Stress_compare_kr={kR}.png')
        fig1.savefig(save_path1, dpi=savefig_dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"本构曲线已保存至: {save_path1}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 9))
    for idx, N in enumerate(N_val):
        R0 = kR * N**0.5
        filepath = f"/home/tyt/project/protein_gel/GB1_results/Multi_chains/N_{int(N)}_M_{M}_test_results/average_curves.csv"
        f_val, r_val, n_val = load_average_curve_data(filepath)
        Lc_val = (N - n_val) * xi_f + n_val* alpha * xi_f
        x_val = r_val / Lc_val
        x_val_theo = x_theory(r_val / R0, N)
        ax2.plot(r_val/R0, x_val, 'o', color=colors[idx], markerfacecolor='none',
                markeredgewidth=2, markersize=8,
                label=f'N={int(N)}', zorder=4)
        ax2.plot(r_val/R0, x_val_theo, '-', color='black', linewidth=2, zorder=5)
        
        # 标签与标题
    ax2.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax2.set_ylabel('End-to-end factor $x$', fontsize=label_fontsize)
    ax2.set_title(f'End-to-end factor vs. strain', 
                  fontsize=title_fontsize, pad=20)

    # 网格
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例：模拟点和理论线合并为一个图例项（通过句柄去重实现）
    handles, labels = ax2.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax2.legend(by_label.values(), by_label.keys(),
              fontsize=legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    ax2.set_xlim(1.0, 16.0)
    ax2.set_ylim(0.5, 0.6)

    ax2.tick_params(axis='both', which='major',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    ax2.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5,
                    labelbottom=False, labelleft=False)
    ax2.minorticks_on()

    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        save_path2 = os.path.join(save_dir, f'k_R={kR}_x_compare.png')
        fig2.savefig(save_path2, dpi=savefig_dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"本构曲线已保存至: {save_path2}")


    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 9))
    for idx, N in enumerate(N_val):
        R0 = kR * N**0.5
        filepath = f"/home/tyt/project/protein_gel/GB1_results/Multi_chains/N_{int(N)}_M_{M}_test_results/average_curves.csv"
        f_val, r_val, n_val = load_average_curve_data(filepath)
        lam_val = r_val / R0
        n_frac = n_val / N
        ax3.plot(lam_val, n_frac, 'o', color=colors[idx], markerfacecolor='none',
                markeredgewidth=2, markersize=8,
                label=f'N={int(N)}', zorder=4)

        # 标签与标题
    ax3.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax3.set_ylabel('Unfolding fraction $n/N$', fontsize=label_fontsize)
    ax3.set_title(f'Unfolding fraction vs. strain', 
                  fontsize=title_fontsize, pad=20)

    # 网格
    ax3.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例：模拟点和理论线合并为一个图例项（通过句柄去重实现）
    handles, labels = ax3.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax3.legend(by_label.values(), by_label.keys(),
              fontsize=legend_fontsize, framealpha=0.9,
              edgecolor='none', loc='best')

    ax3.set_xlim(1.0, 8.0)
    ax3.set_ylim(-0.1, 1.1)

    ax3.tick_params(axis='both', which='major',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    ax3.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True, left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5,
                    labelbottom=False, labelleft=False)
    ax3.minorticks_on()

    for spine in ax3.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        save_path3 = os.path.join(save_dir, f'k_R={kR}_n_compare.png')
        fig3.savefig(save_path3, dpi=savefig_dpi, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"本构曲线已保存至: {save_path3}") 


def main():
    print("=" * 80)
    print("开始生成本构曲线比较图...")
    print("=" * 80)

    data_dir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/R0_N0.5"  # 可修改为你希望的输出路径
    output_dir = data_dir
    create_visualization(save_dir=output_dir)

    print("=" * 80)
    print("本构曲线比较图生成完成。")
    print("=" * 80)

if __name__ == "__main__":
    main()