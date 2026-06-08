"""
用于处理(GB1)8.xls数据，进行可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ================= 样式配置（完全参照您的代码） =================
# 字体路径（若不存在则使用系统默认）
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# 样式变量
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 30
legend_fontsize = 25
legend_title_fontsize = 30

axes_linewidth = 2.5
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 8
ytick_major_size = 8
grid_linewidth = 1.2
grid_alpha = 0.4
lines_linewidth = 3.5
lines_markersize = 12

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = False
ytick_right = False

figure_dpi = 100
savefig_dpi = 300

# 设置全局字体
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

# ================= 辅助函数 =================
def create_output_directory(output_dir):
    """创建输出目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def read_excel_data(file_path, sheet_name):
    """
    读取单个sheet的extension和force数据，转换为浮点数。
    假设第一列为extension，第二列为force。
    自动处理文本类型，无法转换的值将被设为NaN并删除。
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"读取工作表 {sheet_name} 失败: {e}")
        return None, None

    if df.shape[1] < 2:
        print(f"警告: 工作表 {sheet_name} 列数不足2，跳过")
        return None, None

    # 提取前两列并尝试转换为数值
    extension_raw = df.iloc[:, 0]
    force_raw = df.iloc[:, 1]

    extension = pd.to_numeric(extension_raw, errors='coerce')
    force = pd.to_numeric(force_raw, errors='coerce')

    # 删除缺失值
    valid_mask = ~(extension.isna() | force.isna())
    extension = extension[valid_mask].values
    force = force[valid_mask].values

    if len(extension) == 0:
        print(f"警告: 工作表 {sheet_name} 无有效数值数据，跳过")
        return None, None

    return extension, force

def plot_force_extension(extension, force, sheet_name, output_dir):
    """
    为单个sheet绘制力-延伸曲线，并保存图像。
    使用统一的样式配置。
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # 绘制曲线（蓝色实线，参考代码风格）
    ax.plot(extension, force, color='blue', linewidth=lines_linewidth,
            label=f'{sheet_name}')

    # 标签与标题
    ax.set_xlabel('Extension [nm]', fontsize=label_fontsize)
    ax.set_ylabel('Force [pN]', fontsize=label_fontsize)
    ax.set_title(f'Force-Extension Curve: {sheet_name}',
                 fontsize=title_fontsize, pad=20)

    # 刻度与边框设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction,
                   top=xtick_top, right=ytick_right,
                   width=xtick_major_width, size=xtick_major_size,
                   labelsize=tick_fontsize)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction,
                   top=xtick_top, right=ytick_right,
                   width=xtick_major_width*0.75,
                   size=xtick_major_size*0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    # 网格
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)

    # 图例（可选，若不需要可注释）
    ax.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9)

    ax.set_xlim(0.0, 250.0)
    ax.set_ylim(0.0, 500.0)

    # 调整布局并保存
    plt.tight_layout()
    safe_name = sheet_name.replace('/', '_').replace('\\', '_')
    out_file = Path(output_dir) / f'Force_Extension_{safe_name}.png'
    plt.savefig(out_file, dpi=savefig_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"已保存: {out_file}")

# ================= 主程序 =================
def main():
    # ===== 用户需修改以下路径 =====
    excel_path = "/home/tyt/project/Single-chain/opt+R/(GB1)8.xlsx"          # 请替换为实际Excel文件路径
    output_dir = "/home/tyt/project/Single-chain/opt+R/force_extension_plots" # 图像输出目录
    # =============================

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel文件未找到: {excel_path}")

    # 获取所有sheet名称
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names
    except Exception as e:
        raise RuntimeError(f"无法读取Excel文件: {e}")

    if len(sheet_names) != 29:
        print(f"警告: 文件包含 {len(sheet_names)} 个sheet，期望29个")

    # 创建输出目录
    create_output_directory(output_dir)

    # 遍历每个sheet进行处理
    for sheet in sheet_names:
        print(f"正在处理工作表: {sheet}")
        ext, force = read_excel_data(excel_path, sheet)
        if ext is not None and force is not None:
            plot_force_extension(ext, force, sheet, output_dir)
        else:
            print(f"跳过工作表 {sheet}")

    print("\n全部完成！")

if __name__ == "__main__":
    main()