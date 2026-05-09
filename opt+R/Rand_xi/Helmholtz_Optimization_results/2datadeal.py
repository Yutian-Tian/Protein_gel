import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

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
grid_alpha = 0.4          # 网格线透明度（原名 grid_gamma，现统一为 grid_alpha）
lines_linewidth = 5
lines_markersize = 15

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
    'grid.alpha': grid_alpha,          # 修正为 grid.alpha
    'lines.linewidth': lines_linewidth,
    'lines.markersize': lines_markersize,
    'figure.dpi': figure_dpi,
    'savefig.dpi': savefig_dpi,
    'xtick.direction': xtick_direction,
    'ytick.direction': ytick_direction,
    'xtick.top': xtick_top,
    'ytick.right': ytick_right,
})

# ============ 物理参数 ============
xi_f1 = 5.0          # 第一个domain的折叠态长度
k = 2.0              # k = xi_ui/xi_fi
gamma_default = 1.1  # 默认 gamma 值（当文件中无法解析时使用）
beta_default = 0.7   # 默认 beta 值
force_limit = 10.0

def contour_length_Lci(n_i, xi_fi):
    """
    轮廓长度: L_{ci}(n_i) = ξ_fi + n_i (k - 1)ξ_fi
    """
    n_i = np.asarray(n_i, dtype=float)
    return xi_fi + n_i * (k - 1) * xi_fi

def MS_force(r, L_c):
    """
    Marko–Siggia 力（WLC模型近似）
    """
    x = np.asarray(r, dtype=float) / np.asarray(L_c, dtype=float)
    force = np.where(x < 1.0,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def parse_gamma_beta_from_filename(filename):
    """从文件名解析 gamma 和 beta 数值，返回 (gamma_val, beta_val) 或 (None, None)"""
    basename = os.path.basename(filename).replace('.csv', '')
    parts = basename.split('_')
    gamma_val = None
    beta_val = None
    for i, p in enumerate(parts):
        if p == 'gamma' and i+1 < len(parts):
            try:
                gamma_val = float(parts[i+1])
            except ValueError:
                pass
        elif p == 'beta' and i+1 < len(parts):
            try:
                beta_val = float(parts[i+1])
            except ValueError:
                pass
    return gamma_val, beta_val

def plot_n_curves(ax, r, n1, n2, title):
    """绘制n1, n2 vs r曲线"""
    ax.plot(r, n1, color='blue', linewidth=lines_linewidth, label='$n_1$', zorder=3)
    ax.plot(r, n2, color='red', linewidth=lines_linewidth, linestyle='--', label='$n_2$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$n$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)  # 修正为 alpha
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')  # 修正为 framealpha
    # 刻度设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

def plot_force_curves(ax, r, force1, force2, title):
    """绘制f1, f2 vs r曲线"""
    ax.plot(r, force1, color='blue', linewidth=lines_linewidth, label='$f_1$', zorder=3)
    ax.plot(r, force2, color='red', linewidth=lines_linewidth, linestyle='--', label='$f_2$', zorder=3)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$f$', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)  # 修正为 alpha
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')  # 修正为 framealpha
    # 刻度设置
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width, length=xtick_major_size)
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    ax.minorticks_on()
    ax.set_xlim(0, r[-1])
    ax.set_ylim(0, force_limit)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

# ============ 批量绘图主程序 ============
def main():
    data_dir = "/home/tyt/project/Single-chain/opt+R/Rand_xi/Helmholtz_Optimization_results/2-domain_results_E0=5,g2"
    output_dir = os.path.join(data_dir, "Figure")
    os.makedirs(output_dir, exist_ok=True)

    # 匹配文件名格式：gamma_*_beta_*.csv
    file_pattern = os.path.join(data_dir, "gamma_*_beta_*.csv")
    csv_files = glob.glob(file_pattern)

    print(f"找到 {len(csv_files)} 个文件，开始批量绘图...")

    for csv_file in csv_files:
        try:
            # 读取数据（假设有表头，列顺序：r, r1, r2, n1, n2）
            df = pd.read_csv(csv_file, header=0)
            if df.shape[1] < 5:
                print(f"跳过文件 {csv_file}：列数不足5")
                continue

            # 提取数据并确保为浮点类型
            r = df.iloc[:, 0].values.astype(float)
            r1 = df.iloc[:, 1].values.astype(float)
            r2 = df.iloc[:, 2].values.astype(float)
            n1 = df.iloc[:, 3].values.astype(float)
            n2 = df.iloc[:, 4].values.astype(float)

            # 解析 gamma 和 beta
            gamma_val, beta_val = parse_gamma_beta_from_filename(csv_file)
            if gamma_val is None or beta_val is None:
                print(f"警告：文件 {os.path.basename(csv_file)} 中未能解析 gamma/beta，将使用默认值 gamma={gamma_default}, beta={beta_default}")
                gamma_used = gamma_default
                beta_used = beta_default
            else:
                gamma_used = gamma_val
                beta_used = beta_val

            xi_f2 = beta_used * xi_f1

            # 计算轮廓长度和力
            Lc1 = contour_length_Lci(n1, xi_f1)
            Lc2 = contour_length_Lci(n2, xi_f2)
            force1 = MS_force(r1, Lc1)
            force2 = MS_force(r2, Lc2)

            # 将无穷大力替换为 NaN
            force1 = np.where(np.isfinite(force1), force1, np.nan)
            force2 = np.where(np.isfinite(force2), force2, np.nan)

            # 标题
            if gamma_val is not None and beta_val is not None:
                title = f"γ = {gamma_val}, 1/β = {beta_val}"
            else:
                title = os.path.basename(csv_file)

            # ===== 绘制 n 曲线 =====
            fig_n, ax_n = plt.subplots(1, 1, figsize=(12, 9))
            plot_n_curves(ax_n, r, n1, n2, title)
            plt.tight_layout()
            n_output = os.path.join(output_dir, os.path.basename(csv_file).replace('.csv', '.png'))
            plt.savefig(n_output, dpi=savefig_dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig_n)
            print(f"已保存 n 曲线: {n_output}")

            # ===== 绘制 force 曲线 =====
            fig_f, ax_f = plt.subplots(1, 1, figsize=(12, 9))
            plot_force_curves(ax_f, r, force1, force2, title)
            plt.tight_layout()
            f_output = os.path.join(output_dir, os.path.basename(csv_file).replace('.csv', '_force.png'))
            plt.savefig(f_output, dpi=savefig_dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig_f)
            print(f"已保存 force 曲线: {f_output}")

        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    print("批量绘图完成！")

if __name__ == "__main__":
    main()