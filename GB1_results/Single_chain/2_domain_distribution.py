import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.signal import fftconvolve
from tqdm import tqdm

# ============ 字体与绘图风格设置 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 24
label_fontsize = 24
tick_fontsize = 20
legend_fontsize = 18

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 8
ytick_major_size = 8
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 3

figure_dpi = 100
savefig_dpi = 300
xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = True
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
xi_f = 3.6          
mu = 7.6            
Delta_E = 11.7      # 固定一个 Delta_E 用于测试
kB_T = 1.0          # 约化温度，假设 beta = 1

# ===================== 基础物理函数 =====================
def phi(x):
    x = np.clip(x, 0, 0.999999)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def Lc_of_n(n):
    return xi_f * (1 + (mu - 1) * n)

def F_WLC(r, n):
    Lc = Lc_of_n(n)
    x = r / Lc
    x_clipped = np.clip(x, 0, 0.9999)
    res = Lc * phi(x_clipped)
    return np.where(x >= 0.9999, 1e10, res)

def U_n(n, Delta_E):
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

def F_d(r, n, Delta_E):
    return F_WLC(r, n) + U_n(n, Delta_E)

# ===================== 概率密度计算 =====================
def calc_pd_3d(R_grid, n, Delta_E):
    F = F_d(R_grid, n, Delta_E)
    F_min = np.min(F)
    p_3d = np.exp(-(F - F_min) / kB_T)
    p_3d = np.where(F >= 1e9, 0.0, p_3d)
    return p_3d

def calc_pd_1d_for_sampling(r_grid, n, Delta_E):
    F = F_d(r_grid, n, Delta_E)
    F_min = np.min(F)
    p_3d = np.exp(-(F - F_min) / kB_T)
    p_3d = np.where(F >= 1e9, 0.0, p_3d)
    P_r = 4 * np.pi * r_grid**2 * p_3d
    norm = np.sum(P_r)
    if norm > 0:
        P_r = P_r / norm
    else:
        P_r = np.ones_like(P_r) / len(P_r)
    return P_r

# ===================== 空间卷积 (第一步) =====================
def calc_radial_pc_from_fft(fft_pd1, fft_pd2, R_grid, r_bins, dV):
    """从预先计算好的 FFT 结果中计算径向分布 P_c(r)"""
    pc_3d = np.real(np.fft.ifftn(fft_pd1 * fft_pd2))
    pc_3d = np.fft.fftshift(pc_3d)
    
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    Pc_r = np.zeros_like(r_centers)
    
    for i in range(len(r_centers)):
        mask = (R_grid >= r_bins[i]) & (R_grid < r_bins[i+1])
        prob_shell = np.sum(pc_3d[mask]) * dV
        Pc_r[i] = prob_shell / (r_bins[i+1] - r_bins[i])
        
    # 【修复点 1】：加判断，防止除以零
    norm = np.trapezoid(Pc_r, r_centers)
    if norm > 1e-300:
        Pc_r /= norm
    else:
        Pc_r[:] = 0.0  # 尾部无概率，直接置零
        
    return r_centers, Pc_r

def prepare_fft_pd(n_vals, Delta_E, N_grid=64, L_box=80.0):
    """预计算所有 n 值对应的 3D FFT 结果"""
    dx = L_box / N_grid
    x = np.linspace(-L_box/2, L_box/2, N_grid, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R_grid = np.sqrt(X**2 + Y**2 + Z**2)
    dV = dx**3
    
    fft_pd_list = []
    print("预计算三维网格 FFT...")
    for n in tqdm(n_vals, desc="FFT Setup"):
        pd_3d = calc_pd_3d(R_grid, n, Delta_E)
        pd_3d /= np.sum(pd_3d) * dV  # 三维归一化
        fft_pd_list.append(np.fft.fftn(pd_3d))
        
    return fft_pd_list, R_grid, dV

# ===================== 计算 n 的均值和分布 (后续步骤) =====================
def calc_n_average(output_dir, N_n=21, N_grid=64):
    """
    执行后续计算：
    1. 计算所有 n1, n2 组合的 p_c(r, n1, n2)
    2. 提取单变量 p_c(r, n)
    3. 对 n 进行卷积得到 p_ceff(r, n)
    4. 计算 <n>(r)
    """
    print("开始计算 n 的分布与均值...")
    n_vals = np.linspace(0, 1, N_n)
    dn = n_vals[1] - n_vals[0]
    
    # 预计算 FFT
    fft_pd_list, R_grid, dV = prepare_fft_pd(n_vals, Delta_E, N_grid=N_grid, L_box=80.0)
    
    # 定义径向 bin
    r_bins = np.linspace(0, 50.0, 100)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    # 存储 Pc(r, n1, n2) 的矩阵 [N_n, N_n, len(r_centers)]
    Pc_matrix = np.zeros((N_n, N_n, len(r_centers)))
    
    print("计算空间卷积矩阵 Pc(r, n1, n2)...")
    for i in tqdm(range(N_n), desc="Convolution Matrix"):
        for j in range(N_n):
            _, Pc_r = calc_radial_pc_from_fft(
                fft_pd_list[i], fft_pd_list[j], R_grid, r_bins, dV
            )
            Pc_matrix[i, j, :] = Pc_r
            
    # ================= 对 n 进行卷积 =================
    # 为简化，取单变量分布 Pc(r, n) = Pc_matrix[n, n, :] (即 n1=n2=n)
    # 也可以选择 Pc_matrix[n, 0, :]
    Pc_1d = Pc_matrix[np.arange(N_n), np.arange(N_n), :]  # Shape: (N_n, len(r_centers))
    
    Pceff_2d = np.zeros((2*N_n - 1, len(r_centers)))
    n_conv = np.linspace(0, 2, 2*N_n - 1)
    
    for i in range(len(r_centers)):
        # 对 n 维度做一维卷积，并乘以 dn 保持归一化
        Pceff_2d[:, i] = fftconvolve(Pc_1d[:, i], Pc_1d[:, i], mode='full') * dn
        # 重新归一化
        Pceff_2d[:, i] /= np.sum(Pceff_2d[:, i]) * dn
        
    # ================= 计算 <n>(r) =================
    n_avg_r = np.zeros(len(r_centers))
    for i in range(len(r_centers)):
        n_avg_r[i] = np.sum(n_conv * Pceff_2d[:, i]) / np.sum(Pceff_2d[:, i])
        
    # ================= 可视化 =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # 图 1: <n>(r) 随 r 变化
    ax1.plot(r_centers, n_avg_r, '-', color='purple', linewidth=lines_linewidth)
    ax1.set_xlabel('$r$', fontsize=label_fontsize)
    ax1.set_ylabel('$\\langle n \\rangle (r)$', fontsize=label_fontsize)
    ax1.set_title('Average $n$ vs Distance', fontsize=title_fontsize, pad=15)
    ax1.set_xlim(0, 45.0)
    ax1.set_ylim(0, 2.0)
    ax1.grid(True, which="major", ls="--", alpha=grid_alpha)
    
    # 图 2: 热力图展示 Pceff(n|r) 的演化
    # 为了防止 r 点太少导致的横条纹理，可对 r 轴做插值或直接画图
    im = ax2.imshow(Pceff_2d, aspect='auto', origin='lower', 
                    extent=[r_centers[0], r_centers[-1], n_conv[0], n_conv[-1]],
                    cmap='viridis')
    ax2.set_xlabel('$r$', fontsize=label_fontsize)
    ax2.set_ylabel('$n$', fontsize=label_fontsize)
    ax2.set_title('$P_{ceff}(n | r)$', fontsize=title_fontsize, pad=15)
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Probability Density', fontsize=label_fontsize)
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
        ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)
            
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'n_average_vs_r.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")

if __name__ == "__main__":
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Single_chain/2-domain_results' 
    # 测试第一步卷积（如果你已经有此图可以注释掉）
    # test_convolution(output_dir)
    
    # 执行后续步骤：对 n 卷积并计算 <n>(r)
    calc_n_average(output_dir, N_n=21, N_grid=64)