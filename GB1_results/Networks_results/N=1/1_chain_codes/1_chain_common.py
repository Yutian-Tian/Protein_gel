"""
================================================================================
 1-chain 网络模型计算公共模块（1_chain_common.py）

 集中存放一链（连续取向平均）网络的全部物理计算，不包含任何绘图代码：
   · 四种链模型（folded WLC / protein 退火 / unfolded WLC / Gibbs tanh）
     的本构计算
   · 初始模量 G0 与 ξ_f 的扫描计算（完整参考分布，不截断）
   · 参考态长度分布 p0(r) 数据（protein/unfolded/gibbs 用退火分布，
     folded 用自身 WLC 分布）
   · 配套的数值验证函数（方向求积恒等式、高斯链解析解、自由能导数等）

 物理设定（约化单位 kBT=lp=1）：
   · 内部态 n ∈ [0,1]，轮廓长度 Lc(n) = N·ξf·(1 + (μ-1)·n) 严格依赖 n
   · 内部态能量 U(n) = ΔE·n − ΔE·cos(2πn)
   · 单链有效自由能 F_eff(r) = −ln ∫ exp[−Lc·φ(r/Lc) − U(n)] dn（512 点 Gauss）
   · 参考态 p0(r) ∝ r²·exp[−F_eff(r)]
     - 本构曲线：变形前固定截去外侧 TAIL=0.5% 概率（有限参考系综），
       之后不随拉伸删链，保证保留链均满足不可伸长约束
     - 初始模量：使用完整平衡分布，不截断
   · 一链网络应力为连续取向平均 sigma = <r·f·∂g/∂λ>_u，
     初始名义单轴模量 K0 = <r²f' + 4rf>/5（剪切模量为 K0/3）
   · Gibbs（第 17 页）模型共享 protein 参考态，仅替换单链力–拉伸关系：
       r(f) = x(f)·Lc(f)，Lc(f) = N·ξf·[(μ+1)/2 + (μ−1)/2·tanh(K1(f−K2))]
     反演 f_G(r) 后代入同一个连续取向平均积分器

 可视化请使用同目录下的独立脚本：
   1_chain_constitutive_curves.py / 1_chain_modulus_G0_xi_f.py /
   1_chain_reference_distribution.py
================================================================================
"""
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp

try:
    TRAPEZOID = np.trapezoid
except AttributeError:  # NumPy < 1.26 兼容
    TRAPEZOID = np.trapz

# ============ 核心物理参数 ============
N = 1                 # 折叠域数目
XI_F = 3.6            # 单个折叠域轮廓长度 xi_f（单位 lp）
mu = 7.6              # 展开态/折叠态轮廓长度比
Delta_E = 11.9        # 内部态能量尺度（kBT）

# 有限参考系综：变形前固定去除最外侧 TAIL 概率并重新归一化；
# 之后不随拉伸删链。该近似保证保留链均满足不可伸长约束。
TAIL = 0.005
BASE_MARGIN = 0.995   # 常规伸长区上界（占几何上限的比例）
MARGIN = 1.0 - 1e-8   # 终止间隙：算到轮廓剩余 1e-8 处
N_STATE = 512         # 内部态常规 Gauss 求积阶数
N_RADIAL = 256        # 径向常规求积阶数
N_ANGLE = 48          # 方向常规求积阶数

# 三种链模型键名（图例文字、颜色与线型由各可视化脚本自定）
KINDS = ('folded', 'protein', 'unfolded')

# ============ Gibbs（第 17 页）唯象参数 ============
K1 = 6.5      # tanh 斜率参数
K2 = 1.48     # tanh 中心力参数

# 含 Gibbs 的完整模型列表（默认 KINDS 不含 gibbs，旧脚本行为不变）
KINDS_GIBBS = KINDS + ('gibbs',)


# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分 phi(x)=x^2(3-2x)/(4(1-x))。"""
    return x**2 * (3 - 2*x) / (4 * (1 - x))


def f_MS(x):
    """Marko-Siggia 微观力解析式（beta=1, lp=1）。"""
    return 0.25 * (1 - x)**(-2) - 0.25 + x


def fcp(x):
    """Marko-Siggia 微观力对相对伸长 x 的解析导数。"""
    return 0.5 * (1 - x)**(-3) + 1.0


# ===================== Gibbs 唯象模型单链力 =====================
def _gibbs_contour(f, N, xi_f):
    """Gibbs 唯象轮廓长度 L_c(f) 及其导数 L_c'(f)（内部辅助函数）。

    L_c(f)  = N*xi_f*[(mu+1)/2 + (mu-1)/2*tanh(K1(f-K2))]
    L_c'(f) = N*xi_f*(mu-1)*K1/2*(1-tanh^2)
    与 3_chain_common.py 的 GibbsNetwork.contour 公式一致。
    """
    t = np.tanh(K1*(f - K2))
    L = N*xi_f*((mu + 1)/2 + (mu - 1)*t/2)
    Lp = N*xi_f*(mu - 1)*K1*(1 - t*t)/2
    return L, Lp


def gibbs_force_stiffness(r, N=N, xi_f=XI_F):
    """r(f)=x*Lc(fMS(x)) 的反演；返回单链力 f_G(r) 与 df_G/dr。

    签名与 3_chain_common.gibbs_force_stiffness(r, N, xi_f) 一致；
    60 次二分对整数组向量化。r 可为任意形状数组。
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0) or np.any(r >= N*mu*xi_f):
        raise ValueError('gibbs distance must be below maximum contour length')
    lo = np.zeros_like(r)
    hi = np.ones_like(r) - 1e-10
    for _ in range(60):
        x = (lo + hi) / 2
        f = f_MS(x)
        L, _ = _gibbs_contour(f, N, xi_f)
        below = x * L < r
        lo = np.where(below, x, lo)
        hi = np.where(below, hi, x)
    x = (lo + hi) / 2
    f = f_MS(x)
    L, Lp = _gibbs_contour(f, N, xi_f)
    fp = fcp(x) / (L + x*Lp*fcp(x))
    return f, fp


def gibbs_chain_energy(r, N=N, xi_f=XI_F, order=512):
    """F_G(r) = ∫_0^r f_G(s) ds，独立于网络应力积分。

    对应 3_chain_common.GibbsNetwork.chain_energy；
    作为模块级函数接受 (N, xi_f)，供 response 按链类型分派时直接调用。
    """
    r = np.asarray(r, dtype=float)
    t, w = gauss_unit(order)
    # gauss_unit 已把权重映射到 [0,1]，不能再次除以 2。
    return r * np.sum(gibbs_force_stiffness(r[..., None]*t, N, xi_f)[0]*w,
                      axis=-1)


def Lc_of_n(n, N, xi_f):
    """内部状态 n 对应的轮廓长度 Lc(n)=N*xi_f*(1+(mu-1)n)。"""
    return N * xi_f * (1 + n*(mu - 1))


def U_of_n(n):
    """内部变量的势能 U(n)=Delta_E*n-Delta_E*cos(2 pi n)，单位 kBT。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)


@lru_cache(maxsize=None)
def gauss_unit(points):
    """[0,1] 上的 Gauss-Legendre 节点与权重（带缓存）。"""
    q, w = np.polynomial.legendre.leggauss(points)
    return (q + 1) / 2, w / 2


# ===================== 单链自由能与退火统计 =====================
def annealed_statistics(r, N, xi_f, n_points=512):
    """对内部态直接求积的退火统计量。

    返回 (logZ, <f_n>, <f_n'>-Var(f_n), <f_n'>, Var(f_n))，kBT=lp=1。
    每个端距将 n 的积分区间限定为满足 Lc(n)>r 的可达域；
    不封顶链力、不裁掉负刚度（涨落修正可使有效刚度为负）。
    """
    r = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("End distance must be finite and nonnegative")
    # 可达域下界：Lc(n)>r 要求 n > lower
    lower = np.maximum(0., (r / (N * xi_f) - 1) / (mu - 1))
    if np.any(lower >= 1):
        raise ValueError('End distance must remain below maximum contour length')
    t, wn = gauss_unit(n_points)
    # 在每个 r 各自的 [lower,1] 区间上映射 Gauss 节点
    n = lower[..., None] + (1 - lower[..., None]) * t
    Lc = Lc_of_n(n, N, xi_f)
    x = r[..., None] / Lc
    # 对数权重：WLC 自由能 + 内部态势能 + 测度雅可比
    lw = -Lc * phi(x) - U_of_n(n) + np.log((1 - lower[..., None]) * wn)
    logZ = logsumexp(lw, axis=-1)
    conditional = np.exp(lw - logZ[..., None])
    fn = f_MS(x)
    f = np.sum(conditional * fn, axis=-1)
    frozen_fp = np.sum(conditional * fcp(x) / Lc, axis=-1)
    variance = np.sum(conditional * (fn - f[..., None])**2, axis=-1)
    # 退火刚度 = 冻结刚度 - 内部态涨落修正
    return logZ, f, frozen_fp - variance, frozen_fp, variance


def build_annealed(N, xi_f, n_points=512, r_points=1024):
    """在完整支撑 [0, Lmax] 上制备退火参考分布（模量扫描用）。

    采用 r=Lmax*t^2 变换：保留完整支撑，并在小端距区域增加节点。
    """
    t, wt = gauss_unit(r_points)
    Lmax = N * xi_f * mu
    r = Lmax * t**2
    wr = wt * 2 * Lmax * t
    logZ, f, fp, frozen_fp, variance = annealed_statistics(r, N, xi_f, n_points)
    # 径向母权重 w(r)=r^2 Z(r)，归一化为概率质量
    logmass = logZ + 2 * np.log(r) + np.log(wr)
    mass = np.exp(logmass - logsumexp(logmass))
    return dict(r=r, mass=mass, f=f, fp=fp, frozen_fp=frozen_fp,
                variance=variance, N=N, xi_f=xi_f, n_points=n_points)


# ===================== 一链网络响应与参考态求积 =====================
def response(r, xi, kind, order=N_STATE, need_energy=True):
    """单链能量/力/刚度响应。

    kind='protein' : 退火自由能（内部态 n 求积）；
    kind='folded'  : 固定轮廓 L=xi 的 WLC；
    kind='unfolded': 固定轮廓 L=mu*xi 的 WLC；
    kind='gibbs'   : tanh 唯象模型，单链力 f_G(r) 由二分反演给出。

    need_energy=False 时跳过能量计算（gibbs 的 F_G 是额外 512 点积分，
    不免费）；对 protein/WLC 能量与力一起算出，传 False 也不浪费。
    """
    if kind == 'protein':
        logz, f, fp, _, _ = annealed_statistics(r, 1, xi, order)
        return -logz, f, fp
    if kind == 'gibbs':
        f, fp = gibbs_force_stiffness(r, N, xi)
        F = gibbs_chain_energy(r, N, xi, order) if need_energy else None
        return F, f, fp
    L = xi if kind == 'folded' else mu*xi
    x = np.asarray(r)/L
    if np.any(x >= 1) or np.any(x < 0):
        raise ValueError('Chain distance outside contour length')
    return L*phi(x), f_MS(x), fcp(x)/L


def finite_reference(xi, kind, points=N_RADIAL, tail=TAIL):
    """固定截断参考分布：先截去外侧 tail 概率，再生成径向求积节点。

    截断在变形前一次性完成，不随 lambda 删链（有限参考系综约定）。
    kind='gibbs' 时使用与 'protein' 相同的退火参考态。
    """
    L = xi if kind == 'folded' else mu*xi
    # 细网格上求 CDF 分位点，确定最长保留端距 cutoff
    rg = np.linspace(0, L, max(2001, int(L/.005)+1))
    F = response(rg[:-1], xi, 'folded' if kind == 'folded' else 'protein')[0]
    logz = -F
    p = np.append(rg[:-1]**2*np.exp(logz-logz.max()), 0.)
    cdf = cumulative_trapezoid(p, rg, initial=0.)
    cdf /= cdf[-1]
    cutoff = np.interp(1-tail, cdf, rg)
    # 截断支撑内的 Gauss 求积节点（r=cutoff*t^2 集中小端距节点）
    t, wt = gauss_unit(points)
    r, wr = cutoff*t**2, 2*cutoff*t*wt
    logz = -response(r, xi, 'folded' if kind == 'folded' else 'protein')[0]
    lw = 2*np.log(r)+logz+np.log(wr)
    mass = np.exp(lw-logsumexp(lw))
    return dict(r=r, mass=mass, cutoff=cutoff, L=L)


def endpoint_nodes(gap, order=24):
    """按到轮廓端点的距离分几何面板求积，解析接近轮廓时的尖锐贡献。

    返回 (u, w)，其中 u 为到端点距离的补（越接近轮廓 u 越密）。
    """
    edges=np.unique(np.r_[0., np.geomspace(gap*.1,1.,
        max(2,int(np.ceil(-np.log10(gap*.1)))+1))])
    t,w=gauss_unit(order)
    distance=np.concatenate([a+(b-a)*t for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(b-a)*w for a,b in zip(edges[:-1],edges[1:])])
    return 1-distance,weights


def constitutive_grid(pre):
    """本构伸长网格：常规区 120 点 + 轮廓附近按剩余间隙对数加密 20 点。"""
    limit=pre['L']/pre['cutoff']
    base=np.linspace(1.,BASE_MARGIN*limit,120)
    extended=(1-np.geomspace(1-BASE_MARGIN,1-MARGIN,21)[1:])*limit
    return np.r_[base,extended]


def orient_average(pre, xi, kind, lam, angles=N_ANGLE, energy=False,
                   state_order=N_STATE):
    """一链网络连续取向平均：对 u=|cos(theta)| 积分单链响应。

    接近轮廓（gap 小）时对方向与径向同时启用几何分面求积；
    分块调用 response 避免 r×u×n 完整张量常驻内存。
    返回名义应力 sigma/(rho kBT)（energy=True 时返回能量密度）。

    kind='gibbs' 时 response 内部走 gibbs_force_stiffness，
    积分权重、stretch、derivative 与 protein/WLC 完全相同，
    因此是严格的连续取向平均，而非离散链方向平均。
    """
    gap=1-lam*pre['cutoff']/pre['L']
    if gap <= 0:
        raise ValueError('Retained reference support reaches the contour length')
    if gap < 1-BASE_MARGIN-1e-12:
        # 相同的固定参考密度重新求积；没有随拉伸删链。
        v,wv=endpoint_nodes(gap, max(24, angles//2))
        r=pre['cutoff']*v
        reference_kind='folded' if kind=='folded' else 'protein'
        logz=-response(r,xi,reference_kind)[0]
        logmass=2*np.log(r)+logz+np.log(pre['cutoff']*wv)
        pre=dict(pre,r=r,mass=np.exp(logmass-logsumexp(logmass)))
        u,wu=endpoint_nodes(gap,max(24,angles//2))
    else:
        u, wu = gauss_unit(angles)
    # 单轴不可压缩伸长的局部伸长因子 g(u,lambda)
    stretch = np.sqrt(lam**2*u*u+lam**-1*(1-u*u))
    derivative = (lam*u*u-.5*lam**-2*(1-u*u))/stretch
    total = 0.
    # 各块直接积分内部态；避免r×u×n完整张量常驻内存。
    for start in range(0, len(pre['r']), 24):
        r = pre['r'][start:start+24]
        mass = pre['mass'][start:start+24]
        F, f, _ = response(r[:, None]*stretch, xi, kind, order=state_order,
                           need_energy=energy)
        values = F if energy else r[:, None]*f*derivative
        total += np.sum(mass*np.sum(values*wu,axis=-1))
    return total


def initial_modulus(pre, xi, kind):
    """初始名义单轴模量 K0=<r^2 f'+4 r f>/5；剪切模量为 K0/3。"""
    r,w = pre['r'],pre['mass']
    _,f,fp = response(r,xi,kind,need_energy=False)
    return np.sum(w*(r*r*fp+4*r*f))/5


def full_reference(xi, kind):
    """完整（不截断）参考分布：protein/unfolded/gibbs 用退火分布，
    folded 用自身 WLC 分布。"""
    if kind != 'folded':
        return build_annealed(1,xi)
    x,wx = gauss_unit(1024)
    lw = 2*np.log(x)-xi*phi(x)+np.log(wx)
    return dict(r=xi*x,mass=np.exp(lw-logsumexp(lw)))


# ===================== Gibbs 网络类（与 3_chain 接口对齐） =====================
class GibbsNetwork:
    """tanh 唯象 Gibbs 网络（一链连续取向平均）。

    方法名与 3_chain_common.GibbsNetwork 对齐：
    contour / force_stiffness / sigma / E0 / chain_energy / network_energy。
    区别：sigma 走一链连续取向平均 orient_average（三链为离散平均）；
    参考态由 pre dict 提供（三链为 (r, p0) 元组）。
    """

    def __init__(self, reference, N=N, xi_f=XI_F):
        self.pre = reference
        self.r = reference['r']
        self.p0 = reference['mass']
        self.cutoff = reference['cutoff']
        self.N = N
        self.xi_f = xi_f
        self.L = N*mu*xi_f
        self.R0 = np.sqrt(TRAPEZOID(self.r**2 * self.p0, self.r))

    def contour(self, f):
        """L_c(f) 及其导数；与 3_chain_common.GibbsNetwork.contour 一致。"""
        return _gibbs_contour(f, self.N, self.xi_f)

    def force_stiffness(self, r):
        """f_G(r), df_G/dr；与 3_chain_common 同名方法一致。"""
        return gibbs_force_stiffness(r, self.N, self.xi_f)

    def sigma(self, lam, angles=N_ANGLE):
        """一链网络名义应力：连续取向平均，与 protein/unfolded 同款。"""
        return orient_average(self.pre, self.xi_f, 'gibbs', lam, angles=angles)

    def E0(self):
        """初始名义单轴模量 K0；剪切模量为 K0/3。"""
        return initial_modulus(self.pre, self.xi_f, 'gibbs')

    def chain_energy(self, r, order=512):
        """F_G(r)=∫_0^r f_G(s)ds；与 3_chain_common 同名方法一致。"""
        return gibbs_chain_energy(r, self.N, self.xi_f, order)

    def network_energy(self, lam, order=512):
        """一链网络能量：连续取向平均；对 λ 的导数即 sigma(λ)。"""
        return orient_average(self.pre, self.xi_f, 'gibbs', lam,
                              energy=True, state_order=order)


# ===================== 数值验证 =====================
def validate(pre_protein, pre_folded, xi, kinds=KINDS, n_angle=N_ANGLE):
    """一链网络的物理与数值验证；全部断言通过才返回验证文本行。

    检查内容：方向求积恒等式、高斯链解析解、零应力参考态、初始斜率、
    自由能导数、径向/角度求积加密及轮廓附近求积加密。
    kind='gibbs' 时使用 pre_protein（参考态继承 protein），
    上述所有检查（G0 与 dF/dλ）自动适用。
    """
    lines=[]
    # 方向求积权重的一阶、二阶矩（P2 勒让德多项式）
    u,w = gauss_unit(n_angle)
    a=(3*u*u-1)/2
    assert abs(np.sum(w*a))<1e-12
    assert np.isclose(np.sum(w*a*a),.2,atol=1e-12)
    # 高斯网络：<r²>=3，f=r，解析sigma=lambda-lambda^-2，G0=3。
    for lm in [1.,1.5,3.]:
        stretch=np.sqrt(lm*lm*u*u+lm**-1*(1-u*u))
        d=(lm*u*u-.5*lm**-2*(1-u*u))/stretch
        assert np.isclose(3*np.sum(w*stretch*d),lm-lm**-2,atol=1e-12)
    for kind in kinds:
        pre=pre_folded if kind=='folded' else pre_protein
        # 初始模量与 sigma(lambda) 在 lambda=1 处差分斜率一致
        g=initial_modulus(pre,xi,kind)
        h=1e-5
        slope=(orient_average(pre,xi,kind,1+h)-
               orient_average(pre,xi,kind,1-h))/(2*h)
        assert abs(orient_average(pre,xi,kind,1.))<1e-10
        assert np.isclose(g,slope,rtol=2e-6,atol=1e-7),(kind,g,slope)
        # 中程伸长：应力 = 自由能对 lambda 的导数；并与加密方向求积对比
        lm=1+.6*(MARGIN*pre['L']/pre['cutoff']-1)
        stress=orient_average(pre,xi,kind,lm)
        dF=(orient_average(pre,xi,kind,lm+h,energy=True)-
            orient_average(pre,xi,kind,lm-h,energy=True))/(2*h)
        assert np.isclose(stress,dF,rtol=2e-6,atol=1e-7)
        high=orient_average(pre,xi,kind,lm,angles=96)
        assert np.isclose(stress,high,rtol=2e-4,atol=1e-7),(kind,stress,high)
        # 加密径向求积（512 节点）对比
        refined=finite_reference(xi,kind,points=512)
        sr=orient_average(refined,xi,kind,lm,angles=96)
        assert np.isclose(stress,sr,rtol=2e-4,atol=1e-7),(kind,stress,sr)
        # 轮廓末端：常规网格 vs 加密方向求积
        endpoint=MARGIN*pre['L']/pre['cutoff']
        end_sigma=orient_average(pre,xi,kind,endpoint)
        end_refined=orient_average(pre,xi,kind,endpoint,angles=96)
        assert np.isclose(end_sigma,end_refined,rtol=2e-4,atol=1e-7)
        lines.append(f'{kind}: endpoint_sigma={end_sigma:.10g}, endpoint_refined={end_refined:.10g}')
        line=(f'{kind}: G0_finite_reference={g:.10g}, sigma_derivative={slope:.10g}, '
              f'lambda={lm:.6g}, sigma={stress:.10g}, dF={dF:.10g}, refined={sr:.10g}')
        print(line,flush=True);lines.append(line)
    return lines


# ===================== 本构计算 =====================
def compute_constitutive(xi=XI_F, kinds=KINDS, verbose=True):
    """固定截断参考态：各网络本构 sigma(lambda) + 完整数值验证。

    默认 kinds=KINDS 输出三种曲线；传 kinds=KINDS_GIBBS 时额外输出
    Gibbs tanh 曲线（参考态继承 protein，连续取向平均）。
    返回 dict：curves={kind: (lam, sig)}、G0={kind: 初始模量}、
    protein/folded 参考态求积字典、可选 gibbs 网络对象、validation 验证文本行。
    """
    protein = finite_reference(xi, 'protein')
    folded = finite_reference(xi, 'folded')
    gibbs = GibbsNetwork(protein, N=N, xi_f=xi) if 'gibbs' in kinds else None
    validation = validate(protein, folded, xi, kinds=kinds)
    curves = {}
    G0 = {}
    for kind in kinds:
        pre = folded if kind == 'folded' else protein
        lam = constitutive_grid(pre)
        sig = []
        for i, lm in enumerate(lam):
            sig.append(orient_average(pre, xi, kind, lm))
            if verbose and i % 35 == 0:
                print(f'constitutive {kind}: {i+1}/{len(lam)}', flush=True)
        sig = np.asarray(sig)
        G0[kind] = float(initial_modulus(pre, xi, kind))
        curves[kind] = (lam, sig)
    return dict(curves=curves, G0=G0, protein=protein, folded=folded,
                gibbs=gibbs, validation=validation)


# ===================== 模量扫描计算 =====================
def modulus_scan(xi_min=1., xi_max=100., n_points=100, verbose=True):
    """在完整参考分布下扫描初始模量 G0(xi_f)。

    返回 (xis, values)：values 为 (n_xi, 3) 数组，列序
    [folded, protein, gibbs]。Gibbs 按 1.png 使用完整蛋白参考态，
    直接积分 (r² f_G' + 4 r f_G)/5，不从本构曲线拟合。
    """
    xis = np.linspace(xi_min, xi_max, n_points)
    values = []
    for i, xi in enumerate(xis):
        # 两种链模型各自的完整参考分布
        pp = full_reference(xi, 'protein')
        pf = full_reference(xi, 'folded')
        values.append([initial_modulus(pf, xi, 'folded'),
                       initial_modulus(pp, xi, 'protein'),
                       initial_modulus(pp, xi, 'gibbs')])
        if verbose and i % 10 == 0:
            print(f'modulus scan: {i+1}/{len(xis)}', flush=True)
    return xis, np.asarray(values)


# ===================== 参考态分布数据 =====================
def reference_log_normalization(xi, radial_order):
    """完整参考分布 ln∫r²Z(r)dr 的 Gauss 求积（径向 r=contour*t² 变换）。"""
    t, weights = gauss_unit(radial_order)
    contour = N * mu * xi
    r = contour * t**2
    logz = annealed_statistics(r, N, xi, N_STATE)[0]
    return float(logsumexp(2*np.log(r) + logz
                          + np.log(2*contour*t*weights)))


def reference_distribution_data(xi=XI_F):
    """三种网络的完整参考态分布数据（不使用本构的 0.5% 尾截断）。

    返回 (r, p_protein, rf, p_folded)：protein/unfolded/gibbs 共用同一
    退火分布；folded WLC 为自身 WLC 平衡分布。
    """
    # 蛋白/解折叠/Gibbs：母权重 w(r)=r²Z(r) 在 [0, mu*xi] 上归一化
    contour = mu*N*xi
    r = np.linspace(0., contour, max(2001, int(contour/.005)+1))
    logz = annealed_statistics(r[:-1], N, xi, N_STATE)[0]
    lognorm = reference_log_normalization(xi, 1024)
    protein = np.zeros_like(r)
    protein[1:-1] = np.exp(2*np.log(r[1:-1])+logz[1:]-lognorm)
    # 折叠 WLC：自身平衡分布 r²exp(-xi*phi(r/xi)) 在 [0, xi] 上归一化
    rf = np.linspace(0., N*xi, max(2001, int(N*xi/.005)+1))
    x, wx = gauss_unit(1024)
    lf = N*xi
    norm_folded = logsumexp(2*np.log(lf*x)-lf*phi(x)+np.log(lf*wx))
    folded = np.zeros_like(rf)
    folded[1:-1] = np.exp(2*np.log(rf[1:-1])-lf*phi(rf[1:-1]/lf)-norm_folded)
    return r, protein, rf, folded
