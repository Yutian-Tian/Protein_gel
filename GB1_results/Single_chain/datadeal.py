import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
import sys

# parameters for the plot
xi_f = 3.6
alpha = 7.6
N = 10    # domain的数量
f_max = 5.0

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

def PlotMS(L_c, lineType ):
    r = np.linspace(0, 0.95*L_c, 1000)
    x = r/L_c
    x = np.asarray(x)
    L_c = np.asarray(L_c)
    
    # 初始化结果数组
    result = np.zeros_like(x, dtype=float)
    
    # 处理x < 1的情况
    mask = x < 1.0
    if np.any(mask):
        x_masked = x[mask]
        result[mask] = 0.25 * ((1 - x_masked)**(-2) - 1 + 4*x_masked)
    
    # 处理x >= 1的情况
    result[~mask] = np.inf

    line = plt.plot(r, result, lineType, color='purple', linewidth=lines_linewidth, label=f'$L_c = {L_c}$', zorder=3)

    return line

def process_data(r_file, n_file):
    """
    处理r_value.csv和n_values.csv文件
    
    参数:
    r_file: r_value.csv文件路径
    n_file: n_values.csv文件路径
    
    返回:
    f: f值数组
    r_opt: r_opt数组（每行第二列开始求和）
    n_opt: n_opt数组（每行第二列开始求和）
    df_r: 原始r数据DataFrame
    df_n: 原始n数据DataFrame
    """
    try:
        print(f"正在读取文件: {r_file} 和 {n_file}")
        
        # 读取CSV文件，第一行作为列名
        df_r = pd.read_csv(r_file, header=0)
        df_n = pd.read_csv(n_file, header=0)
        
        print(f"r数据形状: {df_r.shape}, n数据形状: {df_n.shape}")
        
        # 检查数据形状是否匹配
        if df_r.shape[0] != df_n.shape[0]:
            print("警告: r和n数据的行数不匹配!")
            min_rows = min(df_r.shape[0], df_n.shape[0])
            df_r = df_r.iloc[:min_rows]
            df_n = df_n.iloc[:min_rows]
            print(f"使用前{min_rows}行数据")
        
        # 提取f列（第一列），从第一行开始（因为第0行是数据，不是表头）
        # 注意：第一行是标签，但pandas已将其作为列名，所以数据从索引0开始
        f = df_r.iloc[:, 0].astype(float).values
        
        # 计算r_opt：对每行从第二列开始求和
        r_data = df_r.iloc[:, 1:].astype(float)
        r_opt = r_data.sum(axis=1).values
        
        # 计算n_opt：对每行从第二列开始求和
        n_data = df_n.iloc[:, 1:].astype(float)
        n_opt = n_data.sum(axis=1).values
        
        print(f"数据处理完成:")
        print(f"  f值数量: {len(f)}")
        print(f"  r_opt数量: {len(r_opt)}")
        print(f"  n_opt数量: {len(n_opt)}")
        
        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"  f范围: [{f.min():.4f}, {f.max():.4f}]")
        print(f"  r_opt范围: [{r_opt.min():.4f}, {r_opt.max():.4f}]")
        print(f"  n_opt范围: [{n_opt.min():.4f}, {n_opt.max():.4f}]")
        
        return f, r_opt, n_opt, df_r, df_n
        
    except FileNotFoundError as e:
        print(f"错误: 未找到文件 - {e}")
        return None, None, None, None, None
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def create_visualizations(f, r_opt, n_opt, save_dir=None):
    """
    创建可视化图表
    
    参数:
    f: f值数组
    r_opt: r_opt数组
    n_opt: n_opt数组
    save_dir: 保存目录路径
    
    返回:
    fig1, fig2: 两个图形对象
    """
    # 确保有保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # ============ 创建第一幅图: f-r_opt ============
    # f是纵轴，r_opt是横轴
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    
    PlotMS(36,"-")
    PlotMS(273.6,"--")
    # 使用散点图绘制f-r_opt关系：r_opt在x轴，f在y轴
    scatter1 = ax1.scatter(r_opt, f, 
                          s=lines_markersize*5,  # 散点大小
                          c='red',                # 散点颜色
                          alpha=0.7,              # 透明度
                          edgecolors='red',     # 边缘颜色
                          linewidths=1,           # 边缘宽度
                          label='Optimization', 
                          zorder=3)
    
    # 设置标签和标题
    ax1.set_xlabel('End-to-end distance $r$', fontsize=label_fontsize)
    ax1.set_ylabel('Force $f$', fontsize=label_fontsize)
    ax1.set_title('$f$ - $r$ Relationship', 
                  fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax1.set_xlim(0.0, 0.95*alpha*N*xi_f)
    ax1.set_ylim(0.0, f_max)
    
    # 设置刻度参数
    ax1.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax1.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax1.minorticks_on()
    
    # 强化边框
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    # 保存第一幅图
    if save_dir:
        save_path1 = os.path.join(save_dir, 'f_r_opt_scatter.png')
        fig1.savefig(save_path1, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"第一幅图(散点图)已保存至: {save_path1}")
    
    # ============ 创建第二幅图: n_opt-f ============
    # f是横轴，n_opt是纵轴
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 9))
    
    # 使用散点图绘制n_opt-f关系：f在x轴，n_opt在y轴
    scatter2 = ax2.scatter(f, n_opt/N, 
                          s=lines_markersize*5,  # 散点大小
                          c='blue',               # 散点颜色
                          alpha=0.7,              # 透明度
                          edgecolors='blue',     # 边缘颜色
                          linewidths=1,           # 边缘宽度
                          label='Optimization', 
                          zorder=3)
    
    # 设置标签和标题
    ax2.set_xlabel('Force $f$', fontsize=label_fontsize)
    ax2.set_ylabel('Unfolded Number $n$', fontsize=label_fontsize)
    ax2.set_title('$n$ - $f$ Relationship', 
                  fontsize=title_fontsize, pad=20)
    
    # 设置网格
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    
    # 设置图例
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, 
               edgecolor='none', loc='best')
    
    # 设置坐标轴范围
    ax2.set_xlim(0.0, f_max)
    ax2.set_ylim(-0.05, 1.05)
    
    # 设置刻度参数
    ax2.tick_params(axis='both', which='major', 
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width,
                    length=xtick_major_size)
    
    ax2.tick_params(axis='both', which='minor',
                    direction=xtick_direction,
                    top=xtick_top,
                    right=ytick_right,
                    bottom=True,
                    left=True,
                    width=xtick_major_width*0.75,
                    length=xtick_major_size*0.5)
    
    # 开启次刻度
    ax2.minorticks_on()
    
    # 强化边框
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    plt.tight_layout()
    
    # 保存第二幅图
    if save_dir:
        save_path2 = os.path.join(save_dir, 'n_opt_f_scatter.png')
        fig2.savefig(save_path2, dpi=savefig_dpi, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print(f"第二幅图(散点图)已保存至: {save_path2}")
    
    return fig1, fig2


def main():
    """主程序"""
    print("=" * 80)
    print("数据处理和可视化程序")
    print("=" * 80)
    
    # ============ 在这里指定文件路径 ============
    # 修改这些路径为您实际的路径
    r_file_path = "/home/tyt/project/protein_gel/GB1_results/Single_chain/results/r_values_unified.csv"
    n_file_path = "/home/tyt/project/protein_gel/GB1_results/Single_chain/results/n_values_unified.csv"
    output_dir = "/home/tyt/project/protein_gel/GB1_results/Single_chain/results"  # 保存结果的目录
    
    # 如果没有命令行参数，使用默认路径
    if len(sys.argv) > 1:
        r_file_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        n_file_path = sys.argv[2]
    
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    print(f"r_value.csv路径: {r_file_path}")
    print(f"n_values.csv路径: {n_file_path}")
    print(f"输出目录: {output_dir}")
    
    # 处理数据
    f, r_opt, n_opt, df_r, df_n = process_data(r_file_path, n_file_path)
    
    if f is None:
        print("数据处理失败，程序退出")
        return
    
    # 创建可视化图表
    print("\n" + "=" * 80)
    print("创建散点图...")
    print("=" * 80)
    
    # 创建单独的两幅图
    fig1, fig2 = create_visualizations(f, r_opt, n_opt, save_dir=output_dir)
    

    
    print("\n" + "=" * 80)
    print("程序执行完毕!")
    print(f"所有散点图已保存至: {os.path.abspath(output_dir)}")
    print("=" * 80)

# ============ 运行主程序 ============
if __name__ == "__main__":
    main()