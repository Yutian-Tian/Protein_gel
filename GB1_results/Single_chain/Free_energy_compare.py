import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid, simpson

# ============ 字体与绘图风格设置 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 28
label_fontsize = 28
tick_fontsize = 24
legend_fontsize = 20
legend_title_fontsize = 24

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 3
lines_markersize = 10

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

# ============ 核心物理参数 ============
xi_f = 3.6
mu = 7.6
Delta_E = 11.9
k1 = 6.5
k2 = 1.50

R_LIMIT = 0.999 * xi_f * mu

# ===================== 基础物理函数 =====================
def phi(x):
    x = np.clip(x, 0, 0.999999)
    return x**2 * (3 - 2*x) / (4 * (1 - x))

def fc(x):
    x = np.clip(x, 0, 0.999999)
    return 0.25 * (1 - x)**(-2) - 0.25 + x

def Lc_of_n(n):
    return xi_f * (1 + (mu - 1) * n)

def F_WLC(r, n):
    Lc = Lc_of_n(n)
    x = r / Lc
    x_clipped = np.clip(x, 0, 0.9999)
    res = Lc * phi(x_clipped)
    return np.where(x >= 0.9999, 1e10, res)

def U_n(n):
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)

def F_d(r, n):
    return F_WLC(r, n) + U_n(n)

# ===================== 计算双分支自由能 =====================
def calc_branches(r):
    n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
    if n_min >= 1.0:
        return np.nan, np.nan, np.nan, np.nan

    F_f, n_f = np.nan, np.nan
    F_u, n_u = np.nan, np.nan

    if n_min < 0.5:
        res_f = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 0.5), method='bounded')
        if res_f.success and np.isfinite(res_f.fun):
            F_f, n_f = res_f.fun, res_f.x

    lower_high = max(n_min, 0.5)
    if lower_high < 1.0:
        res_u = minimize_scalar(lambda n: F_d(r, n), bounds=(lower_high, 1.0), method='bounded')
        if res_u.success and np.isfinite(res_u.fun):
            F_u, n_u = res_u.fun, res_u.x

    return F_f, n_f, F_u, n_u

def calc_delta_F(r_grid):
    delta_F = np.full_like(r_grid, np.nan, dtype=float)
    for i, r in enumerate(r_grid):
        F_f, _, F_u, _ = calc_branches(r)
        if np.isfinite(F_f) and np.isfinite(F_u):
            delta_F[i] = F_u - F_f
    return delta_F

# ===================== 原有方法 =====================
def calc_method1():
    x_vals = np.linspace(0, 0.999, 1000)
    f_vals = fc(x_vals)
    n_vals = 0.5 * (1 + np.tanh(k1 * (f_vals - k2)))
    r_vals = x_vals * Lc_of_n(n_vals)
    F_eff1 = cumulative_trapezoid(f_vals, r_vals, initial=0)
    return r_vals, F_eff1

def calc_method2(r_grid):
    F_eff2 = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            F_eff2[i] = 1e10
            continue
        F_folded = F_d(r, 0.0)
        F_unfolded = F_d(r, 1.0)
        F_min = min(F_folded, F_unfolded)
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff2[i] = 1e10
            continue
        Z_sum = np.exp(-(F_folded - F_min)) + np.exp(-(F_unfolded - F_min))
        Z_sum = max(Z_sum, 1e-300)
        F_eff2[i] = F_min - np.log(Z_sum)
    return F_eff2 + Delta_E

def calc_method3_continuous(r_grid):
    F_eff3 = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            F_eff3[i] = 1e10
            continue
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        res = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 1.0), method='bounded')
        F_min = res.fun
        if F_min > 1e9 or not np.isfinite(F_min):
            F_eff3[i] = 1e10
            continue
        n_vals = np.linspace(0, 1, 1000)
        F_d_vals = F_d(r, n_vals)
        integrand = np.exp(-(F_d_vals - F_min))
        integral = simpson(integrand, x=n_vals)
        integral = max(integral, 1e-300)
        F_eff3[i] = F_min - np.log(integral)
    return F_eff3 - F_eff3[0]

def calc_method4_adiabatic(r_grid):
    F_eff4 = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            F_eff4[i] = 1e10
            continue
        F_f, _, F_u, _ = calc_branches(r)
        candidates = [f for f in [F_f, F_u] if np.isfinite(f)]
        if len(candidates) == 0 or min(candidates) > 1e9:
            F_eff4[i] = 1e10
        else:
            F_eff4[i] = min(candidates)
    return F_eff4 + Delta_E

def calc_n_mean(r_grid):
    n_mean_vals = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            n_mean_vals[i] = 1.0
            continue
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        res = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 1.0), method='bounded')
        F_min = res.fun
        if F_min > 1e9 or not np.isfinite(F_min):
            n_mean_vals[i] = 1.0
            continue
        n_vals = np.linspace(n_min, 1.0, 1000)
        F_d_vals = F_d(r, n_vals)
        weights = np.exp(-(F_d_vals - F_min))
        Z = simpson(weights, x=n_vals)
        Z = max(Z, 1e-300)
        numerator = simpson(n_vals * weights, x=n_vals)
        n_mean_vals[i] = numerator / Z
    return n_mean_vals

def calc_n_mean_2state(r_grid):
    n_mean_vals = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            n_mean_vals[i] = 1.0
            continue
        F_f = F_d(r, 0.0)
        F_u = F_d(r, 1.0)
        candidates = [f for f in [F_f, F_u] if np.isfinite(f)]
        if len(candidates) == 0 or min(candidates) > 1e9:
            n_mean_vals[i] = 1.0
            continue
        delta_F = F_u - F_f
        n_mean_vals[i] = 0.5 * (1.0 - np.tanh(0.5 * delta_F))
    return n_mean_vals

# ===================== 新增：五种自由能对比计算 =====================
def calc_five_free_energies(r_grid):
    """计算图中的五种自由能，新增系综平均 <F>"""
    F_strict = np.zeros_like(r_grid)
    F_adiabatic = np.zeros_like(r_grid)
    F_approx_cont = np.zeros_like(r_grid)
    F_approx_2state = np.zeros_like(r_grid)
    F_ensemble_avg = np.zeros_like(r_grid)  # 第五条曲线
    
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            F_strict[i] = np.nan
            F_adiabatic[i] = np.nan
            F_approx_cont[i] = np.nan
            F_approx_2state[i] = np.nan
            F_ensemble_avg[i] = np.nan
            continue
            
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        
        # 1. 寻找全局极小值 (Adiabatic)
        res = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 1.0), method='bounded')
        F_min = res.fun
        F_adiabatic[i] = F_min
        
        # 计算连续配分函数和平均 n
        n_vals = np.linspace(n_min, 1.0, 1000)
        F_d_vals = F_d(r, n_vals)
        weights = np.exp(-(F_d_vals - F_min))
        
        Z = simpson(weights, x=n_vals)
        Z = max(Z, 1e-300)
        
        # 2. Strict free energy
        F_strict[i] = F_min - np.log(Z)
        
        # 3. Approx free energy (Continuous)
        numerator_n = simpson(n_vals * weights, x=n_vals)
        n_mean_cont = numerator_n / Z
        F_approx_cont[i] = F_d(r, n_mean_cont)
        
        # 4. Approx free energy (2-state)
        F_f = F_d(r, 0.0)
        F_u = F_d(r, 1.0)
        delta_F = F_u - F_f
        n_mean_2state = 0.5 * (1.0 - np.tanh(0.5 * delta_F))
        F_approx_2state[i] = F_d(r, n_mean_2state)
        
        # 5. 系综平均自由能 <F>
        numerator_F = simpson(F_d_vals * weights, x=n_vals)
        F_ensemble_avg[i] = numerator_F / Z
        
    # 减去 r=0 时的初始自由能，进行基准对齐
    F_strict -= F_strict[0]
    F_adiabatic -= F_adiabatic[0]
    F_approx_cont -= F_approx_cont[0]
    F_approx_2state -= F_approx_2state[0]
    F_ensemble_avg -= F_ensemble_avg[0]
    
    return F_strict, F_adiabatic, F_approx_cont, F_approx_2state, F_ensemble_avg

# ===================== 可视化函数 =====================
def plot_comparison(output_dir):
    r_max = 0.98 * xi_f * mu
    r_grid = np.linspace(0, r_max, 1000)

    r_m1, F_m1 = calc_method1()
    F_m2 = calc_method2(r_grid)
    F_m3 = calc_method3_continuous(r_grid)
    F_m4 = calc_method4_adiabatic(r_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_yscale('linear')
    ax1.plot(r_m1, F_m1, '-',  color='blue',   label='"Gibbs" free energy', linewidth=lines_linewidth)
    ax1.plot(r_grid, F_m2, '-',  color='red',    label='Strict free energy(2-state)', linewidth=lines_linewidth)
    ax1.plot(r_grid, F_m3, '-.', color='green',  label='Strict free energy', linewidth=lines_linewidth)
    ax1.plot(r_grid, F_m4, '--', color='orange', label='Adiabatic free energy', linewidth=lines_linewidth)

    ax1.set_xlabel('End-to-end Length $r$', fontsize=label_fontsize)
    ax1.set_ylabel('$F_{\\text{eff}}(r)$', fontsize=label_fontsize)
    ax1.set_title('Effective Free Energy Landscape', fontsize=title_fontsize, pad=25)
    ax1.set_xlim(0, r_max)
    ax1.set_ylim(0.0, 30.0)
    ax1.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='upper left')

    n_star_vals = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r >= R_LIMIT:
            n_star_vals[i] = 1.0
            continue
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1) + 1e-4)
        res = minimize_scalar(lambda n: F_d(r, n), bounds=(n_min, 1.0), method='bounded')
        n_star_vals[i] = res.x if res.fun < 1e9 else 1.0

    n_mean_vals = calc_n_mean(r_grid)
    n_mean_2state = calc_n_mean_2state(r_grid)
    ax2.plot(r_grid, n_star_vals, '-',  color='purple', linewidth=lines_linewidth, label='Minimal $n^*(r)$')
    ax2.plot(r_grid, n_mean_vals, '--', color='darkcyan', linewidth=lines_linewidth, label='Continuous')
    ax2.plot(r_grid, n_mean_2state, '-.', color='orange', linewidth=lines_linewidth, label='2-state')

    ax2.set_xlabel('End-to-End Length $r$', fontsize=label_fontsize)
    ax2.set_ylabel('Unfolding Fraction $\\langle n \\rangle$', fontsize=label_fontsize)
    ax2.set_title('Unfolding Fraction', fontsize=title_fontsize, pad=20)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax2.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
        ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'F_eff_comparison_final.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")

def plot_free_energy_difference_vs_n(output_dir):
    r_list = [2, 4, 6, 10]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_list)))

    for idx, r in enumerate(r_list):
        n_min = max(0.0, (r / xi_f - 1) / (mu - 1))
        if n_min >= 0.999:
            continue
        n_vals = np.linspace(n_min, 1.0, 1000)
        F_d_vals = F_d(r, n_vals)
        F_f, _, F_u, _ = calc_branches(r)
        candidates = [f for f in [F_f, F_u] if np.isfinite(f)]
        F_min = min(candidates) if len(candidates) > 0 else np.min(F_d_vals)
        y_vals = F_d_vals - F_min
        ax.plot(n_vals, y_vals, '-', color=colors[idx], linewidth=lines_linewidth, label=f'$r = {r}$')

    ax.set_xlabel('Unfolding Fraction $n$', fontsize=label_fontsize*0.9)
    ax.set_ylabel('Free Energy Difference $\\beta \\Delta F$', fontsize=label_fontsize*0.9)
    ax.set_title('Free Energy Landscape', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.0, 50.0)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize*0.9, framealpha=0.9, edgecolor='none', loc='upper right')

    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'Free_energy_landscape.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")
    plt.show()

def plot_delta_F_vs_r(output_dir):
    r_max = 0.98 * xi_f * mu
    r_grid = np.linspace(0, r_max, 500)

    delta_F = calc_delta_F(r_grid)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_grid, delta_F, '-', color='crimson', linewidth=lines_linewidth,
            label='$\\Delta F_{u-f}(r) = F_u(r) - F_f(r)$')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    valid_idx = np.where(np.isfinite(delta_F))[0]
    if len(valid_idx) > 0:
        valid_r = r_grid[valid_idx]
        valid_dF = delta_F[valid_idx]
        sign_changes = np.where(np.diff(np.sign(valid_dF)))[0]

        if len(sign_changes) > 0:
            i = sign_changes[0]
            r1, r2 = valid_r[i], valid_r[i+1]
            d1, d2 = valid_dF[i], valid_dF[i+1]
            r_eq = r1 - d1 * (r2 - r1) / (d2 - d1)

            ax.plot(r_eq, 0, 'o', color='blue', markersize=12, zorder=5)
            ax.annotate(f'Branch Crossing\n$r_{{eq}} \\approx {r_eq:.3f}$',
                        xy=(r_eq, 0), xytext=(r_eq + 1.5, 5),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                        fontsize=legend_fontsize, ha='left', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
            print(f"找到一级相变分支交叉点: r_eq = {r_eq:.4f}")
        else:
            print("在给定 r 范围内未找到 ΔF_{u-f} 的零点。")
    else:
        print("计算得到的 delta_F 全为 NaN，请检查参数或 r 范围。")

    ax.set_xlabel('End-to-end Length $r$', fontsize=label_fontsize)
    ax.set_ylabel('$\\Delta F_{u-f}(r)$', fontsize=label_fontsize)
    ax.set_title('Free Energy Difference and Branch Crossing', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(-20, 20)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, edgecolor='none', loc='best')

    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'Delta_F_branch_crossing.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")

def plot_n_mean(output_dir):
    r_max = 0.98 * xi_f * mu
    r_grid = np.linspace(0, r_max, 1000)
    n_mean_vals = calc_n_mean(r_grid)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_grid, n_mean_vals, '-', color='darkcyan', linewidth=lines_linewidth)
    ax.set_xlabel('$r$', fontsize=label_fontsize)
    ax.set_ylabel('$\\langle n \\rangle$', fontsize=label_fontsize)
    ax.set_title('Mean Unfolding Fraction', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, r_max)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)

    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'n_mean_vs_r.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")
    plt.show()

# ===================== 修改：绘制五种自由能对比图 =====================
def plot_five_free_energies(output_dir):
    r_max = 0.98 * xi_f * mu
    r_grid = np.linspace(0, r_max, 1000)
    
    F_strict, F_adiabatic, F_approx_cont, F_approx_2state, F_ensemble = calc_five_free_energies(r_grid)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 比较自由能定义
    ax.plot(r_grid, F_strict, '-', color='green', linewidth=lines_linewidth, 
            label='Strict free energy $\\tilde{F}(r)$')
    ax.plot(r_grid, F_adiabatic, '-', color='orange', linewidth=lines_linewidth, 
            label='Adiabatic free energy $F_{\\text{Ad}}(r)$')
    ax.plot(r_grid, F_approx_cont, '--', color='purple', linewidth=lines_linewidth, 
            label='Continuous$F_{\\text{approx}}(r)$')
    ax.plot(r_grid, F_approx_2state, '--', color='red', linewidth=lines_linewidth, 
            label='2-state $F_{\\text{approx}}(r)$ ')
    ax.plot(r_grid, F_ensemble, '--', color='blue', linewidth=lines_linewidth, 
            label='Ensemble average $\\langle F \\rangle (r)$')
    
    ax.set_xlabel('End-to-End Length $r$', fontsize=label_fontsize*0.9)
    ax.set_ylabel('Free Energy $F(r)$', fontsize=label_fontsize*0.9)
    ax.set_title('Free Energy Comparison', fontsize=title_fontsize, pad=20)
    ax.set_xlim(0, r_max)
    ax.set_ylim(0, 35)
    ax.grid(True, which="major", ls="--", alpha=grid_alpha)
    
    # 因为图例较长，调整一下图例显示
    ax.legend(fontsize=legend_fontsize*0.8, framealpha=0.9, edgecolor='none', loc='best')
    
    ax.tick_params(axis='x', which='major', length=6, direction=xtick_direction, top=xtick_top)
    ax.tick_params(axis='y', which='major', width=ytick_major_width, direction=ytick_direction, right=ytick_right)
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'Five_Free_Energies_Comparison.png')
    fig.savefig(save_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    output_dir = '/home/tyt/project/protein_gel/GB1_results/Single_chain/results'

    # 1. 四种方法对比图（原始多方法对比）
    plot_comparison(output_dir)

    # 2. F_d(r,n) - F_min(r) vs n 图
    plot_free_energy_difference_vs_n(output_dir)

    # 3. ΔF_{u-f}(r) 分支交叉图
    plot_delta_F_vs_r(output_dir)
    
    # 4. 五种自由能对比图（包含新增的系综平均）
    plot_five_free_energies(output_dir)

    # 如需单独绘制 <n>(r) 曲线，取消下面这行的注释
    # plot_n_mean(output_dir)