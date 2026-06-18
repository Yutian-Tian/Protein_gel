import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# ============ 字体路径（如有需要可修改） ============
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
legend_fontsize = 25
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 4
lines_markersize = 15

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = False
ytick_right = False

figure_dpi = 100
savefig_dpi = 300

# ============ 应用全局设置 ============
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = font_family

plt.rcParams.update({
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

def force_MS(x):
    force = 0.25*((1 - x)**(-2) + 4*x -1)
    return force

def main():
    # 生成数据
    x_vals = np.linspace(0.01, 0.99, 1000)
    f_MS = force_MS(x_vals)
    f_asy = 0.25*(1 - x_vals)**(-2)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # 绘制曲线
    ax.plot(f_MS, x_vals, '-', color='blue', linewidth=lines_linewidth, label='Marko-Siggia')
    ax.plot(f_asy, x_vals, '--', color='red', linewidth=lines_linewidth, label='Asymptotic')

    # 设置标签和标题
    ax.set_xlabel('Force $f$', fontsize=label_fontsize)
    ax.set_ylabel('End-to-end factor $x$', fontsize=label_fontsize)
    ax.set_title('Extension-Force Curve', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    # 网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.0)

    # 刻度样式（主刻度 + 次刻度）
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width * 0.75, length=xtick_major_size * 0.5)
    ax.minorticks_on()

    # 边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片（可选，这里保存到当前目录下的 Figure 文件夹）
    output_dir = "/home/tyt/project/Single-chain"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "InverFun_curve.png")
    plt.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"图片已保存至: {save_path}")
\

if __name__ == "__main__":
    main()