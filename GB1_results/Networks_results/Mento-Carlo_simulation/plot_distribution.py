"""
分布函数可视化:不同N值下的p(r; N)分布曲线绘制
严格按照图片原始定义：p(r; N) ∝ exp[-F_c(r; N)]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
from scipy.integrate import cumulative_trapezoid

# ============ 字体与绘图风格设置 ============
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

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 4

figure_dpi = 100
savefig_dpi = 300
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
ytick_left = True
ytick_right = True

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
    'axes.linewidth': axes_linewidth,
    'xtick.major.width': xtick_major_width,
    'ytick.major.width': ytick_major_width,
    'xtick.major.size': xtick_major_size,
    'ytick.major.size': ytick_major_size,
    'grid.linewidth': grid_linewidth,
    'grid.alpha': grid_alpha,
    'lines.linewidth': lines_linewidth,
    'figure.dpi': figure_dpi,
    'savefig.dpi': savefig_dpi,
    'xtick.direction': xtick_direction,
    'ytick.direction': ytick_direction,
    'xtick.top': xtick_top,
    'ytick.right': ytick_right,
})

# ============ 核心物理参数 ============
kR = 2.68          
xi_f = 3.6         
alpha = 7.6        # 对应公式中的 μ
k1 = 6.5
k2 = 1.50

# ===================== 基础物理函数 =====================
def f_MS(x):
    """图片中定义的 f_MS 函数"""
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def Lc_of_f(f, N):
    """图片中定义的 L_c(f; N) 函数"""
    return N * xi_f * (0.5*(alpha + 1) + 0.5*(alpha - 1)*np.tanh(k1*(f - k2)))

def G0_3chain_components(N, x_grid):
    """
    根据图片定义计算 p(r; N) 和 F_c(r; N)
    严格按照原公式：p(r; N) ∝ exp[-F_c(r; N)] （不包含 r^2 空间几何因子）
    """
    f = f_MS(x_grid)
    Lc = Lc_of_f(f, N)
    r = x_grid * Lc
    
    # 计算导数 df/dr
    dr_dx = np.gradient(r, x_grid)
    df_dx = np.gradient(f, x_grid)
    df_dr = df_dx / dr_dx
    
    # 通过积分计算 F_c
    Fc = cumulative_trapezoid(f, r, initial=0)
    
    # 计算概率分布 p（核心修改处：去掉 2*log(r)，仅保留 -Fc）
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = - Fc
        log_p -= np.max(log_p)
        p_un = np.exp(log_p)
        p_un = np.nan_to_num(p_un, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 归一化
    Z = np.trapezoid(p_un, r)
    p = p_un / Z
    
    return r, p

# ===================== 多N值分布函数可视化 =====================
def plot_distribution_multi_N(N_vals, save_dir=None):
    """
    同时计算并绘制 N=1, 2, 10 的 p(r; N) 分布曲线
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    labels = [f'$N = {N}$' for N in N_vals]
    
    # 遍历计算并绘图
    for N, color in zip(N_vals, colors):
        # 构建高精度积分网格（避开奇点 x=1）
        x_grid = np.concatenate([
            np.linspace(0, 0.9, 1000),
            1 - np.logspace(-3, -9, 4000)
        ])
        
        r, p = G0_3chain_components(N, x_grid)
        
        ax.plot(r, p, color=color, linewidth=lines_linewidth, 
                label=labels[N_vals.index(N)])
        
        print(f"N={N}: 峰值位置 r_max = {r[np.argmax(p)]:.2f}, 最大范围 r_end = {r[-1]:.2f}")

    ax.axvline(x=1.95, color='purple', linestyle='--', linewidth=3, label='$r=1.95$')

    # 图形设置
    ax.set_xlabel(r'$r$', fontsize=label_fontsize)
    ax.set_ylabel(r'$p(r| N)$', fontsize=label_fontsize)
    ax.set_title('Probability Distribution $p(r| N)$', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    
    # 保持用户设定的X轴范围
    ax.set_xlim(0, 20.0)
    ax.set_ylim(bottom=0)
    
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=xtick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'distribution_p_N.png')
        fig.savefig(path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"多N分布图已保存至: {path}")


# ===================== 主程序入口 =====================
def main():
    # 输出路径
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Networks_results/Mento-Carlo_simulation'
    
    # 需要绘制的 N 值
    N_vals = [1]
    
    plot_distribution_multi_N(N_vals, save_dir=output_dir)

if __name__ == "__main__":
    main()