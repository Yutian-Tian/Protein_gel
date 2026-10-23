"""Three-chain WLC network with a reference-state rupture cutoff.

Model assumptions
-----------------
1. Fixed contour length Lc = N * xi_f.
2. Reference-state radial density is proportional to
       x**2 * exp[-(Lc/lp) * phi(x)], x = r/Lc.
3. At a given lambda, only reference states satisfying lambda*x <= 1-delta
   are load-bearing. Thus x_cut(lambda) = (1-delta)/lambda for lambda >= 1.
4. The reference distribution is normalized once on its original support
   0 <= x < 1, and is NOT renormalized at each lambda. States above the
   cutoff contribute zero stress; the shrinking domain lowers the load-bearing
   chain fraction. Consequently, rho denotes the original chain population.

The returned stress is the reduced nominal stress P/(rho*kB*T). It is the
sum of elastic forces carried by surviving chains, not the derivative of a
truncated free energy with a moving integration boundary. A thermodynamic
free-energy derivative would additionally contain a boundary/damage term.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============ Typography and plotting style (based on the supplied script) ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 30
legend_fontsize = 20
legend_title_fontsize = 30

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 3
lines_markersize = 8

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

# ============ Physical and numerical parameters ============
xi_f = 3.6          # contour length per domain (same length unit as lp)
lp = 1.0            # persistence length; lp=1 reproduces the dimensionless form
N_values = [1, 2, 4, 6, 8, 10]
delta = 0.01         # rupture threshold: chain breaks when x_current >= 1-delta
lambda_max = 10.0
n_lambda = 181
quadrature_points_x = 300


def phi(x):
    """Dimensionless Marko-Siggia WLC free-energy function."""
    x = np.asarray(x, dtype=float)
    return x**2 * (3.0 - 2.0 * x) / (4.0 * (1.0 - x))


def f_MS(x):
    """Dimensionless Marko-Siggia force; caller must ensure 0 <= x < 1."""
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0.0) or np.any(x >= 1.0):
        raise ValueError("f_MS requires finite x in [0, 1); check the rupture cutoff.")
    return 0.25 / (1.0 - x)**2 - 0.25 + x


def gauss_legendre_interval(n_points, lower, upper):
    """Gauss-Legendre nodes and weights on [lower, upper]."""
    if not upper > lower:
        raise ValueError("Integration upper bound must exceed lower bound.")
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    x = 0.5 * (upper - lower) * (nodes + 1.0) + lower
    w = 0.5 * (upper - lower) * weights
    return x, w


def reference_partition_x(Lc, lp, n_points=400):
    """Compute the original reference-state partition integral on 0 <= x < 1.

    The reference distribution is normalized once on its original support.
    The lambda-dependent rupture cutoff is applied only to the stress integral;
    the reference distribution is not renormalized after chains break.
    """
    x_ref, w_ref = gauss_legendre_interval(n_points, 0.0, 1.0)
    a = Lc / lp
    q_ref = x_ref**2 * np.exp(-a * phi(x_ref))
    Z0 = np.sum(w_ref * q_ref)
    if not np.isfinite(Z0) or Z0 <= 0.0:
        raise FloatingPointError("Reference-state partition integral is invalid.")
    return Z0


def calculate_stress_three_chain_survival(
    lambda_vals,
    N,
    xi_f=3.6,
    lp=1.0,
    delta=0.01,
    x_points=300,
):
    """Three-chain nominal stress with brittle deletion of over-stretched chains.

    Uses a common survival cutoff based on the maximally stretched (axial)
    three-chain strand: x <= (1-delta)/lambda. This implements the user's
    stated rule that a reference configuration is excluded once its axial
    strand reaches the rupture threshold.

    Returns
    -------
    stress_reduced : array
        P/(rho*k_B*T), dimensionless/reduced nominal stress.
    survival_fraction : array
        Fraction of the original reference-state chain population still
        load-bearing. This is NOT used to renormalize stress.
    """
    lam_vals = np.asarray(lambda_vals, dtype=float)
    if np.any(lam_vals < 1.0):
        raise ValueError("This rupture rule is written for tensile stretches lambda >= 1.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    if N <= 0.0 or xi_f <= 0.0 or lp <= 0.0:
        raise ValueError("N, xi_f, and lp must be positive.")

    Lc = N * xi_f
    a = Lc / lp
    Z0 = reference_partition_x(Lc, lp, n_points=max(400, x_points),)
    x0 = 1.0 - delta

    stress_reduced = np.zeros_like(lam_vals)
    survival_fraction = np.zeros_like(lam_vals)

    # In three-chain topology: Lambda_x=lambda, Lambda_y=Lambda_z=lambda^(-1/2).
    for i, lam in enumerate(lam_vals):
        x_cut = x0 / lam

        # Integrate on [0, x_cut] with fresh quadrature nodes. The reference
        # density denominator stays fixed at its lambda=1 value.
        x, wx = gauss_legendre_interval(x_points, 0.0, x_cut)
        q = x**2 * np.exp(-a * phi(x))
        p0_density = q / Z0

        # Fraction of the original reference-state population still intact.
        # It is close to, but not mathematically forced to, 1 at lambda=1.
        survival_fraction[i] = np.sum(wx * p0_density)

        # The cutoff guarantees lambda*x <= 1-delta and lambda^(-1/2)*x < 1.
        x_axial = lam * x
        x_transverse = lam**(-0.5) * x
        if np.max(x_axial) >= 1.0:
            raise RuntimeError("Rupture cutoff failed to keep axial stretch below 1.")

        f_axial = f_MS(x_axial)
        f_transverse = f_MS(x_transverse)

        # d(beta F)/d lambda for three-chain topology:
        # (Lc/lp)/3 * x * [ f(lambda*x) - lambda^(-3/2) f(lambda^(-1/2)*x) ]
        chain_response = (a / 3.0) * x * (
            f_axial - lam**(-1.5) * f_transverse
        )

        # Use the initially normalized reference population. Do NOT divide by
        # survival_fraction; broken chains contribute zero force.
        stress_reduced[i] = np.sum(wx * p0_density * chain_response)

    return stress_reduced, survival_fraction


def plot_results(save_dir=None):
    lambda_vals = np.linspace(1.0, lambda_max, n_lambda)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(N_values)))

    stress_curves = {}
    survival_curves = {}
    for N in N_values:
        stress, survival = calculate_stress_three_chain_survival(
            lambda_vals=lambda_vals,
            N=N,
            xi_f=xi_f,
            lp=lp,
            delta=delta,
            x_points=quadrature_points_x,
        )
        stress_curves[N] = stress
        survival_curves[N] = survival

    # Figure 1: Constitutive curves
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    for idx, N in enumerate(N_values):
        ax1.plot(
            lambda_vals,
            stress_curves[N],
            '-',
            color=colors[idx],
            linewidth=lines_linewidth,
            marker='o',
            markevery=15,
            markersize=lines_markersize,
            markerfacecolor='none',
            markeredgewidth=1.5,
            label=f'N={N}',
        )

    ax1.set_xlabel('Stretch ratio $\\lambda$', fontsize=label_fontsize)
    ax1.set_ylabel('Nominal stress $P/(\\rho k_B T)$', fontsize=label_fontsize)
    ax1.set_title(
        f'Three-chain WLC with rupture cutoff ($\\delta={delta:g}$)',
        fontsize=title_fontsize,
        pad=20,
    )
    ax1.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax1.legend(fontsize=legend_fontsize, framealpha=1.0, edgecolor='none', loc='best')
    ax1.set_xlim(1.0, lambda_max)
    ax1.tick_params(
        axis='both', which='major', direction=xtick_direction,
        top=xtick_top, right=ytick_right, bottom=True, left=True,
        width=xtick_major_width, length=xtick_major_size,
    )
    ax1.minorticks_on()
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)
    fig1.tight_layout()

    # Figure 2: Fraction of intact load-bearing chains
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    for idx, N in enumerate(N_values):
        ax2.plot(
            lambda_vals,
            survival_curves[N],
            '-',
            color=colors[idx],
            linewidth=lines_linewidth,
            marker='o',
            markevery=15,
            markersize=lines_markersize,
            markerfacecolor='none',
            markeredgewidth=1.5,
            label=f'N={N}',
        )

    ax2.set_xlabel('Stretch ratio $\\lambda$', fontsize=label_fontsize)
    ax2.set_ylabel('Surviving chain fraction', fontsize=label_fontsize)
    ax2.set_title(
        f'Survival probability ($\\delta={delta:g}$)',
        fontsize=title_fontsize,
        pad=20,
    )
    ax2.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax2.legend(fontsize=legend_fontsize, framealpha=1.0, edgecolor='none', loc='best')
    ax2.set_xlim(1.0, lambda_max)
    ax2.set_ylim(0.0, 1.02)
    ax2.tick_params(
        axis='both', which='major', direction=xtick_direction,
        top=xtick_top, right=ytick_right, bottom=True, left=True,
        width=xtick_major_width, length=xtick_major_size,
    )
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)
    fig2.tight_layout()

    # Figure 3: sensitivity to the rupture threshold delta for N=1
    delta_values = [0.005, 0.01, 0.02, 0.05]
    colors_delta = plt.cm.tab10(np.linspace(0.0, 1.0, len(delta_values)))
    fig3, ax3 = plt.subplots(figsize=(12, 9))
    for idx, delta_i in enumerate(delta_values):
        stress_i, _ = calculate_stress_three_chain_survival(
            lambda_vals=lambda_vals,
            N=1,
            xi_f=xi_f,
            lp=lp,
            delta=delta_i,
            x_points=quadrature_points_x,
        )
        ax3.plot(
            lambda_vals, stress_i, '-', color=colors_delta[idx],
            linewidth=lines_linewidth, marker='o', markevery=15,
            markersize=lines_markersize, markerfacecolor='none',
            markeredgewidth=1.5, label=fr'$\delta={delta_i:g}$'
        )

    ax3.set_xlabel('Stretch ratio $\\lambda$', fontsize=label_fontsize)
    ax3.set_ylabel('Nominal stress $P/(\\rho k_B T)$', fontsize=label_fontsize)
    ax3.set_title('Rupture-threshold sensitivity ($N=1$)', fontsize=title_fontsize, pad=20)
    ax3.grid(True, alpha=grid_alpha, linestyle=':', linewidth=grid_linewidth)
    ax3.legend(fontsize=legend_fontsize, framealpha=1.0, edgecolor='none', loc='best')
    ax3.set_xlim(1.0, lambda_max)
    ax3.tick_params(
        axis='both', which='major', direction=xtick_direction,
        top=xtick_top, right=ytick_right, bottom=True, left=True,
        width=xtick_major_width, length=xtick_major_size,
    )
    ax3.minorticks_on()
    for spine in ax3.spines.values():
        spine.set_linewidth(axes_linewidth)
    fig3.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path1 = os.path.join(save_dir, 'three_chain_WLC_rupture_stress.png')
        path2 = os.path.join(save_dir, 'three_chain_WLC_survival_fraction.png')
        path3 = os.path.join(save_dir, 'three_chain_WLC_delta_sensitivity.png')
        fig1.savefig(path1, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        fig2.savefig(path2, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        fig3.savefig(path3, dpi=savefig_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'Saved constitutive curves: {path1}')
        print(f'Saved survival curves: {path2}')
        print(f'Saved delta sensitivity: {path3}')

    plt.show()


if __name__ == '__main__':
    
    output_dir = "/home/tyt/project/protein_gel/GB1_results/Networks_results/N=1/Figures"

    plot_results(save_dir=output_dir)
