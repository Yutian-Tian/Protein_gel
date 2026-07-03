"""
代码用于分析初始模量随 domain 数量 (N) 的变化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
import sys
from scipy.signal import savgol_filter

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
legend_fontsize = 20
legend_title_fontsize = 35

# 线宽和尺寸
axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 4
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
xi_f = 3.6          # 折叠态持续长度
alpha = 7.6         # 解折叠系数

E_mean = 11.9       # 平均能量差
E_std = 1.7         # 能量差的标准差

# ===================== 【重要变量改动】 =====================
N_val = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # domain 的数量（与代码一相同）
R0_fixed = 1.5                           # 本次分析固定为 1.5
# ===========================================================

M = 300
k1 = 6.5
k2 = 1.50
Rtheo_points = 200  
lambda_max = 300.0  # 最大伸长比

# ===================== 核心函数修改（增加 N 参数） =====================
def Lc(f, N):
    """
    物理含义为：施加外力f时的平均轮廓长度
    """
    contour_length =  N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))
    return contour_length

def MSforce(x):
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def load_average_curve_data(file_path):
    """
    此函数保留不变，但在本次 G0~N 绘制中不会被调用（按照要求无需读取）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径。")

    data = pd.read_csv(file_path)
    if len(data.columns) < 2:
        raise ValueError("CSV文件必须至少包含两列数据。")
    f_val = data.iloc[:, 0].values
    r_val = data.iloc[:, 1].values
    return f_val, r_val

def ModulusMS(R0, N):
    # 使用Marko-Siggia绘制
    x_MS = np.linspace(0.01, 0.99, 3000)
    f_MS = MSforce(x_MS)
    r_MS = x_MS * Lc(f_MS, N)            # 传入 N
    lambda_, stress_ = StressOptimization(R0, N, r_MS, f_MS)
    modulus_ = np.gradient(stress_, lambda_)

    return lambda_, modulus_

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

    # 计算应力：σ = Re [ F'(λRe) - λ^{-3/2} F'(λ^{-1/2}Re) ]
    sigma = Re * (f1 - lambda_ ** (-1.5) * f2)

    # 确保包含 λ=1 且 σ=0
    eps = 1e-12
    if len(lambda_) == 0:
        return lambda_, sigma

    # 检查第一个点（即最小 λ）是否接近 1
    if np.abs(lambda_[0] - 1.0) > eps:
        # 不包含 λ=1，插入 (1, 0) 到数组开头
        lambda_ = np.concatenate(([1.0], lambda_))
        sigma = np.concatenate(([0.0], sigma))
    else:
        # 已包含 λ≈1，将该点应力精确设为 0
        sigma[0] = 0.0
    return lambda_, sigma

def compute_modulus_center(lambda_, sigma):
    # 保持不变
    lambda_ = np.asarray(lambda_)
    sigma = np.asarray(sigma)
    if len(lambda_) < 2:
        raise ValueError("数据点过少，无法进行数值微分")
    modulus = np.zeros_like(sigma)
    n = len(lambda_)
    for i in range(1, n - 1):
        d_lambda = lambda_[i+1] - lambda_[i-1]
        d_sigma = sigma[i+1] - sigma[i-1]
        modulus[i] = d_sigma / d_lambda
    modulus[0] = (sigma[1] - sigma[0]) / (lambda_[1] - lambda_[0])
    modulus[-1] = (sigma[-1] - sigma[-2]) / (lambda_[-1] - lambda_[-2])
    return lambda_, modulus

def initial_modulus(R0, N, fit_points=5):
    """
    计算给定 R0 和 N 下的初始模量 G0 = dσ/dλ|_(λ=1)
    """
    # 生成单链力-伸长关系（Marko-Siggia）
    x_MS = np.linspace(0.01, 0.99, 3000)
    f_MS = MSforce(x_MS)
    r_MS = x_MS * Lc(f_MS, N)              # 传入 N

    # 得到网络应力-拉伸曲线（已包含 λ=1, σ=0）
    lambda_, sigma = StressOptimization(R0, N, r_MS, f_MS)

    # 取前 fit_points 个点（包含 λ=1）
    lam_fit = lambda_[:fit_points]
    sig_fit = sigma[:fit_points]

    # 拟合 σ = G0 * (λ - 1)，无截距
    A = (lam_fit - 1.0).reshape(-1, 1)
    G0, _, _, _ = np.linalg.lstsq(A, sig_fit, rcond=None)
    return G0[0]

def calculate_G0(R0, N):
    """
    理论计算 G0 的值。
    """
    L_c = N * xi_f                         # 使用传入的 N 计算总轮廓长度
    x0 = R0 / L_c
    
    # 数学上的安全检查：分母不能为0
    if x0 >= 1:
        raise ValueError(f"计算错误：x0 = {x0} 必须小于 1。请检查 R0 是否小于总轮廓长度 L_c。")
    
    term1 = x0 / (2 * (1 - x0)**3)
    term2 = 1 / (4 * (1 - x0)**2)
    term3 = 2 * x0
    bracket = term1 + term2 + term3 - 0.25
    
    # 外部系数
    coeff = 1.5 * R0
    
    G0 = coeff * bracket
    return G0

# ===================== 新增：专门用于绘制 G0 ~ N 的函数 =====================
def plot_G0_vs_N(N_val, R0, save_dir=None):
    """
    绘制初始模量 G0 随 N 的变化曲线并保存。
    """
    G0_val = [initial_modulus(R0, n) for n in N_val]  # 计算数值优化拟合解

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 绘制拟合点
    ax.plot(N_val, G0_val, 'o', markerfacecolor='none', label='Optimization', markeredgewidth=2, markersize=15, zorder=4)

    # 绘制理论曲线（连续）
    N_theo = np.logspace(np.log10(0.5), np.log10(N_val[-1] + 2.0), Rtheo_points)
    G0_theo = [calculate_G0(R0, n) for n in N_theo]
    ax.plot(N_theo, G0_theo, '-', linewidth=lines_linewidth, label='Theoretical', alpha=0.8, zorder=5)

    # 可选：添加参考线用于标度分析，例如 G0 ∝ 1/N 或 G0 ∝ 1/N^2
    # 假设起点对齐
    ref_x = [N_theo[0], N_theo[-1]]
    # 与理论曲线终点重合： y = k / x, k = G_end * N_end => y_start = G_end * (N_end / N_start)
    ref_y = [G0_theo[-1] * (N_theo[-1] / N_theo[0]), G0_theo[-1]]
    ax.plot(ref_x, ref_y, '--', color='#666666', linewidth=2.5, label=r'$G_0 \propto N^{-1}$', zorder=3)

    ax.set_xlabel('Number of domains $N$', fontsize=label_fontsize)
    ax.set_ylabel('Initial modulus $G_0$', fontsize=label_fontsize)
    ax.set_title(f'Initial modulus $G_0$ vs $N$ ($R_0={R0:.1f}$)', fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    ax.set_xlim(0.5, N_val[-1] + 2.0)
    ax.set_ylim(0.01, 100.0)

    # 坐标轴美化
    ax.tick_params(axis='both', which='major', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width,
                   length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width*0.75,
                   length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f'G0_vs_N_R0={R0}.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"G0 vs N 曲线已保存至: {path}")

# ========================== 原可视化函数 (适度优化) ==========================
def create_visualization(save_dir=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 原代码里循环 R0_val，这里修改为循环 N_val，展示不同 N 下 G 与 lambda 的关系图
    for N in N_val:
        lambda_, modulus = ModulusMS(R0_fixed, N)  
        ax.scatter(lambda_, modulus, label=f'$N={N:.0f}$', s=lines_markersize, alpha=0.8)

    ax.set_xlabel('Stretch ratio $\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Modulus $G / \\rho k_B T$', fontsize=label_fontsize)
    ax.set_title(f'Modulus $G=\partial \sigma / \partial \lambda$ ($R_0={R0_fixed:.1f}$)', 
                  fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax.legend(fontsize=legend_fontsize, framealpha=0.8, edgecolor='none', loc='best')
    
    ax.set_xlim(1.01, 10.0)  # 适度缩小范围，因为N不同，长度变化快
    ax.set_ylim(0.001, 1000.0)
    
    ax.tick_params(axis='both', which='major', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    if save_dir:
        save_path3 = os.path.join(save_dir, 'modulus_compare_N.png')
        fig.savefig(save_path3, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"Modulus vs Lambda 曲线已保存至: {save_path3}")

# ========================== 主函数 ==========================
def main():
    print("=" * 80)
    print("开始生成初始模量 G0 随 N 变化的比较图...")
    print("=" * 80)
    
    # 输出路径
    data_dir = "/home/tyt/project/protein_gel/GB1_results/Multi_chains"
    output_dir = data_dir  # 保存结果的目录
    
    # 1. 画出模量~λ 的变化 (展示不同 N 的情况)
    create_visualization(save_dir=output_dir)

    # 2. 重点：画出 G0 ~ N 的关系 (R0固定为1.5)
    print("\n开始生成 G0 随 N 变化曲线...")
    plot_G0_vs_N(N_val, R0_fixed, save_dir=output_dir)
    
    print("=" * 80)
    print("所有曲线生成完成。")
    print("=" * 80)

if __name__ == "__main__":
    main()