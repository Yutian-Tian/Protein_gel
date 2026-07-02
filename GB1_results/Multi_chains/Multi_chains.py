import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Iterable, Sequence, Callable
from functools import partial
import numpy as np
from scipy.stats import truncnorm

# ==================== 不可变参数定义 ====================
@dataclass(frozen=True)
class SimParams:
    """模拟参数，全局只读配置"""
    xi_f: float = 3.6
    alpha: float = 7.6
    E_mean: float = 11.9
    E_std: float = 1.7
    E_delta: float = 5.0
    N_domains: int = 10        # 每条链 domain 数量
    M_chains: int = 300        # 链总数
    n_grid: int = 20
    f_grid_initial: int = 5000
    f_max: float = 10.0
    refinement_threshold: float = 0.1
    max_refinement_level: int = 10
    tolerance: float = 1e-8

    @property
    def upper_bound(self) -> float:
        return self.E_mean + self.E_delta

    @property
    def lower_bound(self) -> float:
        return self.E_mean - self.E_delta

# ==================== 不可变结果类型 ====================
@dataclass(frozen=True)
class OptResult:
    """单个 (f, ΔE) 下的优化结果"""
    r_best: float
    n_best: float
    Fd_min: float
    x_best: float

@dataclass(frozen=True)
class DomainOptResult:
    """单个 domain 在自适应力网格上的完整结果"""
    f_vals: Tuple[float, ...]
    r_opt: Tuple[float, ...]
    n_opt: Tuple[float, ...]
    Fd_min: Tuple[float, ...]
    x_opt: Tuple[float, ...]

@dataclass(frozen=True)
class DomainResult:
    """单个 domain 的完整信息（含 ΔE 及自适应结果）"""
    domain_idx: int
    DeltaE: float
    adaptive_result: DomainOptResult

@dataclass(frozen=True)
class ChainResult:
    """单条链完整优化结果"""
    chain_idx: int
    DeltaE: Tuple[float, ...]                         # 长度 N_domains
    unified_f_vals: Tuple[float, ...]                 # 统一力网格
    unified_r_opt: Tuple[Tuple[float, ...], ...]      # N_domains × n_f
    unified_n_opt: Tuple[Tuple[float, ...], ...]
    unified_Fd_min: Tuple[Tuple[float, ...], ...]
    unified_x_opt: Tuple[Tuple[float, ...], ...]
    domain_results: Tuple[DomainResult, ...]

# ==================== 纯函数：物理模型 ====================
def energy_term_U(n: np.ndarray, DeltaE: float) -> np.ndarray:
    return DeltaE * n - DeltaE * np.cos(2 * np.pi * n)

def contour_length_Lci(n: np.ndarray, params: SimParams) -> np.ndarray:
    xi_u = params.alpha * params.xi_f
    return params.xi_f + n * (xi_u - params.xi_f)

def end_to_end_factor_x(r: np.ndarray, n: np.ndarray, params: SimParams) -> np.ndarray:
    L = contour_length_Lci(n, params)
    return r / L

def WLC_free_energy(x: np.ndarray, L: np.ndarray) -> np.ndarray:
    # 超过0.999视为无穷大，对应边界
    safe = x < 0.999
    result = np.where(safe, 0.25 * L * (x**2 * (3.0 - 2.0 * x) / (1.0 - x)), np.inf)
    return result

def MSforce(x: np.ndarray) -> np.ndarray:
    force = np.where(x < 0.99,
                     0.25 * ((1 - x) ** (-2) - 1 + 4 * x),
                     np.inf)
    return force

def single_domain_free_energy(x: np.ndarray, n: np.ndarray, DeltaE: float, 
                              f_ext: float, params: SimParams) -> np.ndarray:
    L = contour_length_Lci(n, params)
    F_wlc = WLC_free_energy(x, L)
    U = energy_term_U(n, DeltaE)
    work = f_ext * x * L
    return F_wlc + U - work

# ==================== 单点优化（内部网格迭代） ====================
def solve_x_from_f(f: float, tol: float = 1e-12) -> float:
    """二分法求解 f(x)=f 的根 x ∈ [0, 1)"""
    if f <= 0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-15          # 避免除零
    # 确保 hi 对应 f(hi) > f
    while MSforce(np.array([hi]))[0] < f:
        hi = 1.0 - (1.0 - hi) * 0.5    # 逐渐靠近 1
    for _ in range(100):               # 二分法迭代
        mid = (lo + hi) / 2
        fmid = MSforce(np.array([mid]))[0]
        if fmid < f:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2

def optimize_single_point(f: float, DeltaE: float, params: SimParams) -> OptResult:
    """
    给定 f 和 ΔE，利用解析 r = x(f)·Lc(n) 将优化降为一维 n 搜索。
    返回 OptResult。
    """
    # 1. 由 f 解出 x
    x_root = solve_x_from_f(f)

    # 2. 一维自适应网格搜索最优 n
    n_min, n_max = 0.0, 1.0
    n_step = (n_max - n_min) / (params.n_grid - 1)

    best_n, best_F = 0.0, np.inf

    while True:
        n_vals = np.linspace(n_min, n_max, params.n_grid)

        # 计算各 n 对应的 Lc, r, 以及自由能
        Lc_vals = contour_length_Lci(n_vals, params)
        # 注意：这里不再需要 WLC 的自由能，可直接用原函数 single_domain_free_energy
        # 但显式写出更高效，也可直接调用 single_domain_free_energy(x_root, n_vals, DeltaE, f, params)
        Fd = single_domain_free_energy(x_root, n_vals, DeltaE, f, params)
        idx = np.argmin(Fd)
        min_Fd = Fd[idx]
        best_n = n_vals[idx]

        # 收敛判断
        if n_step <= params.tolerance:
            break

        # 缩小搜索范围
        n_min = max(0.0, best_n - n_step)
        n_max = min(1.0, best_n + n_step)
        n_step = (n_max - n_min) / 10.0      # 与原逻辑一致，每次细化 10 个点

    # 3. 计算最优 r 和对应的 x_best（x 仅由 f 决定）
    best_Lc = contour_length_Lci(np.array([best_n]), params)[0]
    best_r = x_root * best_Lc

    return OptResult(r_best=float(best_r),
                     n_best=float(best_n),
                     Fd_min=float(min_Fd),
                     x_best=float(x_root))

# ==================== 力值自适应细化 ====================
def detect_transition_regions(f_vals: Sequence[float], n_opt: Sequence[float], 
                              threshold: float = 0.5) -> List[Tuple[float, float]]:
    """检测n值突变的力区域，返回[(f_left, f_right), ...]"""
    regions = []
    for i in range(len(f_vals) - 1):
        if abs(n_opt[i+1] - n_opt[i]) >= threshold:
            regions.append((f_vals[i], f_vals[i+1]))
    return regions

def refine_force_grid(f_vals: Sequence[float], n_opt: Sequence[float], 
                      params: SimParams) -> Tuple[float, ...]:
    """在突变区域插入中点，返回新的有序力值元组"""
    existing = set(f_vals)
    regions = detect_transition_regions(f_vals, n_opt, params.refinement_threshold)
    for f_left, f_right in regions:
        f_mid = (f_left + f_right) / 2.0
        # 避免重复插入
        if not any(math.isclose(f_mid, ef, abs_tol=params.tolerance) for ef in existing):
            existing.add(f_mid)
    return tuple(sorted(existing))

# ==================== 单个 domain 自适应优化 ====================
def optimize_single_domain_adaptive(DeltaE: float, params: SimParams) -> DomainOptResult:
    """对一个domain在自适应力网格上优化，返回完整结果"""
    # 初始力网格
    f_vals = tuple(np.linspace(0.0, params.f_max, params.f_grid_initial))
    # 初始全量计算
    results = tuple(optimize_single_point(f, DeltaE, params) for f in f_vals)
    r_opt = tuple(res.r_best for res in results)
    n_opt = tuple(res.n_best for res in results)
    Fd_min = tuple(res.Fd_min for res in results)
    x_opt = tuple(res.x_best for res in results)

    for level in range(params.max_refinement_level):
        refined_f = refine_force_grid(f_vals, n_opt, params)
        if len(refined_f) == len(f_vals):
            break

        # 检查最小力值间隔
        if len(refined_f) > 1:
            diffs = np.diff(refined_f)
            if np.min(diffs) <= params.tolerance:
                break

        # 筛选新增点
        old_set = set(f_vals)
        new_f_points = tuple(f for f in refined_f 
                             if not any(math.isclose(f, of, abs_tol=1e-10) for of in old_set))
        if not new_f_points:
            break

        # 仅计算新点
        new_results = tuple(optimize_single_point(f, DeltaE, params) for f in new_f_points)

        # 合并所有点并排序（通过字典构建，然后按键排序）
        all_data = {f: (r, n, fd, x) for f, r, n, fd, x in zip(f_vals, r_opt, n_opt, Fd_min, x_opt)}
        for f, res in zip(new_f_points, new_results):
            all_data[f] = (res.r_best, res.n_best, res.Fd_min, res.x_best)

        sorted_f = tuple(sorted(all_data.keys()))
        r_opt = tuple(all_data[f][0] for f in sorted_f)
        n_opt = tuple(all_data[f][1] for f in sorted_f)
        Fd_min = tuple(all_data[f][2] for f in sorted_f)
        x_opt = tuple(all_data[f][3] for f in sorted_f)
        f_vals = sorted_f

        # 再次检查精度
        if len(f_vals) > 1:
            diffs = np.diff(f_vals)
            if np.min(diffs) <= params.tolerance:
                break

    return DomainOptResult(f_vals, r_opt, n_opt, Fd_min, x_opt)

# ==================== ΔE 采样 ====================
def generate_DeltaE(n_samples: int, params: SimParams, seed: int = None) -> Tuple[float, ...]:
    """从截断正态分布采样n_samples个ΔE值"""
    rng = np.random.default_rng(seed)
    a = (params.lower_bound - params.E_mean) / params.E_std
    b = (params.upper_bound - params.E_mean) / params.E_std
    dist = truncnorm(a, b, loc=params.E_mean, scale=params.E_std)
    samples = dist.rvs(size=n_samples, random_state=rng)
    return tuple(samples)

# ==================== 单条链全流程 ====================
def optimize_single_chain(chain_idx: int, params: SimParams, seed: int = None) -> ChainResult:
    """对一条链的所有domain进行优化，返回ChainResult"""
    N = params.N_domains
    # 生成该链的ΔE
    deltaE = generate_DeltaE(N, params, seed)
    
    # 逐个domain自适应优化
    domain_adaptive_results = tuple(
        DomainResult(
            domain_idx=i,
            DeltaE=deltaE[i],
            adaptive_result=optimize_single_domain_adaptive(deltaE[i], params)
        )
        for i in range(N)
    )
    
    # 构建统一力网格（所有domain力值点的并集）
    all_f_sets = (set(dr.adaptive_result.f_vals) for dr in domain_adaptive_results)
    unified_f_vals = tuple(sorted(set().union(*all_f_sets)))
    
    # 在统一网格上重新计算每个domain
    def compute_domain_unified(dr: DomainResult) -> Tuple[Tuple[OptResult, ...], ...]:
        results = tuple(optimize_single_point(f, dr.DeltaE, params) for f in unified_f_vals)
        return results

    # 所有domain的统一结果
    all_unified = tuple(compute_domain_unified(dr) for dr in domain_adaptive_results)
    
    # 转置为按物理量组织 (n_f x N_domains) -> 按 domain 存储为 (N_domains x n_f)
    n_f = len(unified_f_vals)
    unified_r = tuple(tuple(all_unified[i][j].r_best for j in range(n_f)) for i in range(N))
    unified_n = tuple(tuple(all_unified[i][j].n_best for j in range(n_f)) for i in range(N))
    unified_Fd = tuple(tuple(all_unified[i][j].Fd_min for j in range(n_f)) for i in range(N))
    unified_x = tuple(tuple(all_unified[i][j].x_best for j in range(n_f)) for i in range(N))

    return ChainResult(
        chain_idx=chain_idx,
        DeltaE=deltaE,
        unified_f_vals=unified_f_vals,
        unified_r_opt=unified_r,
        unified_n_opt=unified_n,
        unified_Fd_min=unified_Fd,
        unified_x_opt=unified_x,
        domain_results=domain_adaptive_results
    )

# ==================== 文件保存（纯I/O函数） ====================
def save_vector_csv(data: Sequence[float], filepath: Path) -> None:
    """保存一维数组到CSV，逗号分隔"""
    with open(filepath, 'w') as f:
        f.write(','.join(f'{v:.15e}' for v in data) + '\n')

def save_matrix_csv(data: Sequence[Sequence[float]], f_vals: Sequence[float], 
                    filepath: Path, n_domains: int) -> None:
    """保存矩阵（力值行 x domain列）"""
    with open(filepath, 'w') as f:
        # 标题行
        header = ',' + ','.join(f'Domain_{i+1}' for i in range(n_domains)) + '\n'
        f.write(header)
        # 数据行
        for j, fv in enumerate(f_vals):
            row = f'{fv:.15e}'
            for i in range(n_domains):
                row += f',{data[i][j]:.15e}'
            f.write(row + '\n')

def save_chain_results(chain: ChainResult, save_dir: Path, params: SimParams) -> None:
    """保存单条链的所有结果文件"""
    prefix = save_dir / f'chain_{chain.chain_idx + 1}'
    n_dom = params.N_domains
    f_vals = chain.unified_f_vals
    save_matrix_csv(chain.unified_r_opt, f_vals, prefix.with_name(prefix.name + '_r_values_unified.csv'), n_dom)
    save_matrix_csv(chain.unified_n_opt, f_vals, prefix.with_name(prefix.name + '_n_values_unified.csv'), n_dom)
    save_matrix_csv(chain.unified_Fd_min, f_vals, prefix.with_name(prefix.name + '_Fd_values_unified.csv'), n_dom)
    save_matrix_csv(chain.unified_x_opt, f_vals, prefix.with_name(prefix.name + '_x_values_unified.csv'), n_dom)

def save_all_DeltaE(all_chain_DeltaE: Sequence[Sequence[float]], filepath: Path) -> None:
    """保存所有链的ΔE矩阵（M x N）"""
    with open(filepath, 'w') as f:
        n_domains = len(all_chain_DeltaE[0]) if all_chain_DeltaE else 0
        f.write(',' + ','.join(f'Domain_{j+1}' for j in range(n_domains)) + '\n')
        for i, chain_dE in enumerate(all_chain_DeltaE):
            line = f'Chain_{i+1}'
            line += ',' + ','.join(f'{e:.15e}' for e in chain_dE)
            f.write(line + '\n')

def save_params(params: SimParams, filepath: Path) -> None:
    """保存参数文件"""
    with open(filepath, 'w') as f:
        f.write('Parameter,Value,Description\n')
        f.write(f'xi_f,{params.xi_f},折叠态长度\n')
        f.write(f'alpha,{params.alpha},解折叠长度系数\n')
        f.write(f'E_mean,{params.E_mean},解折叠能均值\n')
        f.write(f'E_std,{params.E_std},解折叠能标准差\n')
        f.write(f'E_delta,{params.E_delta},能量截断半宽\n')
        f.write(f'N_domains_per_chain,{params.N_domains},每条链domain数量\n')
        f.write(f'N_chains,{params.M_chains},链数量\n')
        f.write(f'n_grid,{params.n_grid},n方向网格数\n')
        f.write(f'f_grid_initial,{params.f_grid_initial},初始力网格数\n')
        f.write(f'f_max,{params.f_max},最大力值\n')
        f.write(f'refinement_threshold,{params.refinement_threshold},力细化阈值\n')
        f.write(f'max_refinement_level,{params.max_refinement_level},最大细化层数\n')
        f.write(f'tolerance,{params.tolerance},最小力间隔\n')
        f.write(f'upper_bound,{params.upper_bound},ΔE上界\n')
        f.write(f'lower_bound,{params.lower_bound},ΔE下界\n')

def save_statistics(all_chains: Sequence[ChainResult], all_DeltaE: Sequence[Sequence[float]], 
                    filepath: Path, params: SimParams) -> None:
    """保存统计信息"""
    M = len(all_chains)
    N = params.N_domains
    chain_points = [len(ch.unified_f_vals) for ch in all_chains]
    mean_points = np.mean(chain_points)
    std_points = np.std(chain_points, ddof=1) if M > 1 else 0.0
    min_points = min(chain_points)
    max_points = max(chain_points)

    all_flat = [e for chain in all_DeltaE for e in chain]
    mean_dE = np.mean(all_flat)
    std_dE = np.std(all_flat, ddof=1)
    min_dE = min(all_flat)
    max_dE = max(all_flat)

    with open(filepath, 'w') as f:
        f.write('Statistic,Value\n')
        f.write(f'总链数,{M}\n')
        f.write(f'每条链domain数,{N}\n')
        f.write(f'平均力值点数,{mean_points}\n')
        f.write(f'最小力值点数,{min_points}\n')
        f.write(f'最大力值点数,{max_points}\n')
        f.write(f'力值点数标准差,{std_points}\n')
        f.write(f'平均ΔE,{mean_dE}\n')
        f.write(f'ΔE标准差,{std_dE}\n')
        f.write(f'最小ΔE,{min_dE}\n')
        f.write(f'最大ΔE,{max_dE}\n')

# ==================== 主流程（纯组合） ====================
def run_simulation(params: SimParams = SimParams(), 
                   save_dir: Path = None,
                   base_seed: int = 42) -> Tuple[ChainResult, ...]:
    """执行全模拟，返回所有链的结果元组，并保存文件"""
    if save_dir is None:
        save_dir = Path(f'./N_{params.N_domains}_M_{params.M_chains}_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"开始模拟 {params.M_chains} 条链，每条链 {params.N_domains} 个domain")
    print(f"ξ_f: {params.xi_f}")
    print(f"ΔE分布: 均值={params.E_mean}, 标准差={params.E_std}, 截断=[{params.lower_bound}, {params.upper_bound}]")
    print(f"力值范围: [0, {params.f_max}]")
    print(f"保存路径: {save_dir}")
    print(f"{'='*60}")

    # 为每条链生成不同的随机种子
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31-1, size=params.M_chains)

    # 使用map顺序执行（Python map是惰性的，这里显式转为tuple以触发计算）
    # 如果希望并行，可以考虑 concurrent.futures，但为保持函数式简洁，此处顺序执行
    all_chain_results = tuple(
        optimize_single_chain(i, params, int(seeds[i]))
        for i in range(params.M_chains)
    )

    # 提取所有ΔE用于保存
    all_chain_DeltaE = tuple(ch.DeltaE for ch in all_chain_results)

    # 保存
    print("\n保存结果...")
    for chain in all_chain_results:
        save_chain_results(chain, save_dir, params)
    save_all_DeltaE(all_chain_DeltaE, save_dir / 'all_chains_DeltaE.csv')
    save_params(params, save_dir / 'simulation_parameters.csv')
    save_statistics(all_chain_results, all_chain_DeltaE, save_dir / 'simulation_statistics.csv', params)

    print("\n模拟完成！")
    return all_chain_results

# ==================== 入口 ====================
if __name__ == '__main__':
    # 使用默认参数运行（可在此处修改）
    params = SimParams(
        xi_f=3.6,
        alpha=7.6,
        E_mean=11.9,
        E_std=1.7,
        E_delta=5.0,
        N_domains=10,
        M_chains=300,      # 若需快速测试可减小，例如 10
        n_grid=20,
        f_grid_initial=5000,
        f_max=10.0,
        refinement_threshold=0.1,
        max_refinement_level=10,
        tolerance=1e-8
    )
    # 运行模拟，结果保存到当前目录下的指定文件夹
    results = run_simulation(params, base_seed=42)