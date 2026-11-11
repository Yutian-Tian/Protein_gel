#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同质8-domain串联WLC单链的准静态平衡解折叠计算。

算法：全部 2^N 个 0/1 顶点作为初始构型；默认根据完全同质的
置换对称性将其压缩为 N+1 类，然后用逐坐标的自适应导数根二分搜索
(扫描所有驻点及边界) + 坐标下降，选择多初值中的最低自由能。

这是非凸问题的多初值数值搜索，并非全局最优的数学证明。
依赖：numpy、pandas、matplotlib。
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============== 物理参数 ===============
N = 8
XI_F = 15.0
ALPHA = 2.0
DELTA_E = 5.0
A = XI_F * (ALPHA - 1.0)    # dLc/dn

# =============== 数值与输出设置 ===============
N_R = 500
R_MAX_FRACTION = 0.95
CONTOUR_MARGIN = 1e-8  # x = r/Ltotal <= 1-CONTOUR_MARGIN
MAX_SWEEPS = 30
N_TOL = 1e-9
ENERGY_TOL = 1e-10
BISECT_TOL = 1e-12
# 枚举所有2**8=256个0/1初值；同质模型下，置换等价的初值
# 归并为9个独立类别，大幅节约时间。设为False则真的运行256次。
USE_SYMMETRY_REDUCTION = True
OUTDIR = Path("homogeneous_8domain_results")


def phi(x):
    """Marko-Siggia WLC 无量纲Helmholtz自由能密度。只在0<=x<1定义。"""
    x = np.asarray(x, dtype=float)
    return x * x * (3.0 - 2.0*x) / (4.0 * (1.0-x))


def f_ms(x):
    """phi'(x)，单位采用原模型的归一化力。"""
    x = np.asarray(x, dtype=float)
    return 0.25 / (1.0-x)**2 - 0.25 + x


def total_length(n):
    return N * XI_F + A * float(np.sum(n))


def free_energy(r, n):
    """原始自由能，参考态(r=0,n=0)为 -N*DELTA_E。"""
    L = total_length(n)
    x = r / L
    if x >= 1.0 - CONTOUR_MARGIN + 1e-12:
        # 最终返回值应只针对可行状态；容差允许刚好位于约束边界。
        if x > 1.0 - CONTOUR_MARGIN + 2e-12:
            return np.inf
    return float(L*phi(x) + DELTA_E*np.sum(n - np.cos(2*np.pi*n)))


def one_coord_energy(z, L_rest, r):
    """除该坐标外的势能与z无关，无需计入坐标比较。"""
    L = L_rest + A*z
    x = r/L
    return L*phi(x) + DELTA_E*(z - np.cos(2*np.pi*z))


def one_coord_grad(z, L_rest, r):
    """固定其余n_j时，对n_i的精确偏导。支持数组z。"""
    L = L_rest + A*z
    x = r/L
    return A*(phi(x) - x*f_ms(x)) + DELTA_E*(1.0 + 2*np.pi*np.sin(2*np.pi*z))


def derivative_bisection(lo, hi, L_rest, r):
    """对包围的导数零点做标准二分，不是对非凸能量直接二分。"""
    f_lo = float(one_coord_grad(lo, L_rest, r))
    f_hi = float(one_coord_grad(hi, L_rest, r))
    if abs(f_lo) < 1e-12:
        return lo
    if abs(f_hi) < 1e-12:
        return hi
    if f_lo * f_hi > 0:
        raise ValueError("驻点二分需要导数异号的区间")
    for _ in range(60):
        mid = 0.5*(lo+hi)
        f_mid = float(one_coord_grad(mid, L_rest, r))
        if hi-lo <= BISECT_TOL or abs(f_mid) < 1e-12:
            return mid
        if np.sign(f_lo) == np.sign(f_mid):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def best_coordinate_value(r, n, i):
    """第i个n的[可行下界,1]上的自适应二分搜索。

    在16/32/64/128等分网格上检测所有导数变号区间，
    对每个驻点用二分法精确求根，再比较边界与各驻点能量。
    连续两层结果一致时可提前停止（至少检查32等分）。
    """
    L_rest = N*XI_F + A*(float(np.sum(n))-float(n[i]))
    if A <= 0:
        raise ValueError("此实现要求ALPHA>1")

    # 显式执行物理可行性约束 x<=1-CONTOUR_MARGIN。
    lower = max(0.0, (r/(1.0-CONTOUR_MARGIN)-L_rest)/A)
    if lower > 1.0 + 1e-10:
        raise ValueError("当前其它domain不足以满足总长度约束")
    lower = min(lower, 1.0)
    if 1.0-lower <= 1e-14:
        return 1.0

    best_value = float(np.clip(n[i], lower, 1.0))
    best_energy = float(one_coord_energy(best_value, L_rest, r))
    previous = None

    for bins in (16, 32, 64, 128):
        grid = np.linspace(lower, 1.0, bins+1)
        grads = one_coord_grad(grid, L_rest, r)
        candidates = [lower, 1.0]
        # 所有存在导数符号改变的区间都有一个驻点。
        for j in range(bins):
            gj, gnext = float(grads[j]), float(grads[j+1])
            if abs(gj) < 1e-11:
                candidates.append(float(grid[j]))
            if gj * gnext < 0:
                candidates.append(derivative_bisection(
                    float(grid[j]), float(grid[j+1]), L_rest, r
                ))
        if abs(float(grads[-1])) < 1e-11:
            candidates.append(1.0)

        z = np.asarray(candidates)
        E = one_coord_energy(z, L_rest, r)
        idx = int(np.argmin(E))
        z_best = float(z[idx])
        E_best = float(E[idx])

        nstationary = len(candidates)-2
        if E_best < best_energy:
            best_value, best_energy = z_best, E_best

        if previous is not None and bins >= 32:
            old_z, old_E, old_stationary = previous
            if (abs(z_best-old_z) < 2e-9
                and abs(E_best-old_E) < 1e-9
                and nstationary == old_stationary):
                break
        previous = (z_best, E_best, nstationary)
    return best_value


def optimize_from_seed(r, seed):
    """从一个纯0/1初值出发的坐标下降。

    若初值在高r处不可行，先将所需的n从0增加到恰好可行，
    然后再进行逐坐标精确优化。
    """
    n = np.array(seed, dtype=float, copy=True)
    min_sum = max(0.0, (r/(1.0-CONTOUR_MARGIN)-N*XI_F)/A)
    if np.sum(n) < min_sum:
        # 仅在初始状态不满足轮廓长度约束时，构造可行修复。
        needed = min_sum - np.sum(n) + 1e-10
        for j in range(N):
            step = min(1.0-n[j], needed)
            n[j] += step
            needed -= step
            if needed <= 0:
                break
    if total_length(n) < r/(1-CONTOUR_MARGIN)-1e-9:
        raise RuntimeError("无法修复初值：总轮廓长度仍不足")

    for _ in range(MAX_SWEEPS):
        before = n.copy()
        for i in range(N):
            n[i] = best_coordinate_value(r, n, i)
        # 排序只代表同质domain的重标号，不改变物理构型自由能。
        n.sort()
        if np.max(np.abs(n-before)) < N_TOL:
            break
    return n, free_energy(r, n)


def binary_baseline(r):
    """两态n_i=0/1基线：可解析枚举k=0,...,8。"""
    energies = np.full(N+1, np.inf)
    for k in range(N+1):
        L = N*XI_F + A*k
        if r/L <= 1-CONTOUR_MARGIN:
            energies[k] = L*phi(r/L)+DELTA_E*(k-N)
    k = int(np.argmin(energies))
    L = N*XI_F + A*k
    return k, float(energies[k]), float(f_ms(r/L))


def make_binary_starts():
    """严格枚举2^N种组合，每个n_i初始仅取0或1。"""
    all_seeds = [np.array(bits, dtype=float)
                 for bits in itertools.product((0., 1.), repeat=N)]
    if USE_SYMMETRY_REDUCTION:
        # 8个完全相同的domain：所有k个1的排列彼此等价。
        # 仍枚举所有256种顶点，仅实际求解其中9个不同等价类。
        unique = {}
        for seed in all_seeds:
            key = int(np.sum(seed))
            if key not in unique:
                unique[key] = np.sort(seed)
        starts = [unique[k] for k in sorted(unique)]
    else:
        starts = all_seeds
    return all_seeds, starts


def solve_at_r(r, starts):
    best_n, best_E = None, np.inf
    for seed in starts:
        n, E = optimize_from_seed(r, seed)
        if np.isfinite(E) and E < best_E:
            best_n, best_E = n.copy(), E
    if best_n is None:
        raise RuntimeError(f"r={r}: 所有初值均失败")
    # 按解折叠程度从高到低标号；同质domain的物理编号并不固定。
    best_n = np.sort(best_n)[::-1]
    Lc = XI_F*(1+(ALPHA-1)*best_n)
    L = float(np.sum(Lc))
    ri = r*Lc/L
    f = float(f_ms(r/L))
    return best_n, ri, L, best_E, f


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_seeds, starts = make_binary_starts()
    print(f"初始0/1顶点：{len(all_seeds)}；实际独立起点：{len(starts)}")
    print(f"参数：N={N}, xi_f={XI_F}, alpha={ALPHA}, DeltaE={DELTA_E}")

    r_limit = N*XI_F*ALPHA
    r_values = np.linspace(0.0, R_MAX_FRACTION*r_limit, N_R)
    rows = []
    for j, r in enumerate(r_values):
        n, ri, L, E, f = solve_at_r(float(r), starts)
        k2, E2, f2 = binary_baseline(float(r))
        row = {
            "r": r,
            "F_equilibrium": E,
            "deltaF_equilibrium": E+N*DELTA_E,
            "force": f,
            "L_total": L,
            "S": float(np.sum(n)),
            "mean_n": float(np.mean(n)),
            "variance_n": float(np.var(n)),
            "k_binary": k2,
            "F_binary": E2,
            "force_binary": f2,
            "F_binary_minus_continuous": E2-E,
        }
        for i in range(N):
            row[f"n{i+1}"] = n[i]
            row[f"r{i+1}"] = ri[i]
            row[f"f{i+1}"] = f # 同张力，8条力曲线严格相同
        rows.append(row)
        if (j+1) % 25 == 0 or j == 0 or (j+1) == len(r_values):
            print(f"计算进度 {j+1:4d}/{len(r_values)}, r={r:.4f}, "
                  f"S={row['S']:.4f}, f={f:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR/"homogeneous_8domain.csv", index=False)

    plt.figure(figsize=(9,6))
    for i in range(N):
        plt.plot(df.r, df[f"n{i+1}"], lw=1.3, label=f"Sorted $n_{{{i+1}}}$")
    plt.plot(df.r, df.mean_n, "k--", lw=2.4, label="Mean unfolding")
    plt.plot(df.r, df.k_binary/N, ":", lw=1.8, label="Binary baseline")
    plt.xlabel("Total extension r")
    plt.ylabel("Unfolding fraction")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTDIR/"unfolding.png", dpi=250)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(df.r, df.force, lw=2, label="Continuous equilibrium")
    plt.plot(df.r, df.force_binary, "--", lw=1.5, label="Binary baseline")
    plt.xlabel("Total extension r")
    plt.ylabel("Common chain tension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR/"force.png", dpi=250)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(df.r, df.deltaF_equilibrium, lw=2, label="Continuous")
    plt.plot(df.r, df.F_binary+N*DELTA_E, "--", lw=1.5, label="Binary")
    plt.xlabel("Total extension r")
    plt.ylabel("Free energy relative to r=0, n=0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR/"free_energy.png", dpi=250)
    plt.close()

    plt.figure(figsize=(9,6))
    plt.plot(df.r, df.variance_n, lw=2)
    plt.xlabel("Total extension r")
    plt.ylabel("Variance of unfolding fractions")
    plt.tight_layout()
    plt.savefig(OUTDIR/"unfolding_variance.png", dpi=250)
    plt.close()

    if np.nanmin(df.F_binary_minus_continuous) < -1e-7:
        print("警告：连续自由能高于二态基线，需加强优化/检查可行域。")
    else:
        print("检查通过：连续模型自由能不高于二态基线。")
    print(f"输出文件夹：{OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
