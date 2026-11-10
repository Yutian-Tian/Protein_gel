"""
================================================================================
 3-chain 网络模型计算公共模块（3_chain_common.py）

 集中存放三链仿射网络的全部物理计算，不包含任何绘图代码：
   · N=1 四模型本构计算：protein（退火）/ WLC folded / WLC unfolded / Gibbs
   · 初始模量 G0 与 ξ_f 的扫描计算（退火、WLC、Gibbs 三方案）
   · 参考态分布：径向分布 p0(r) 与内部态边缘分布 p(n)
   · 配套的数值验证函数（解析模量 vs 应力差分、分部积分恒等式等）

 物理设定（kBT = lp = 1）：
   · 内部态 n ∈ [0,1]，轮廓长度 Lc(n) = N·ξf·(1 + (μ-1)·n) 严格依赖 n
   · 内部态能量 U(n) = ΔE·n − ΔE·cos(2πn)
   · 单链有效自由能 F_eff(r) = −ln ∫ exp[−Lc·φ(r/Lc) − U(n)] dn（512 点 Gauss）
   · 参考态 p0(r) ∝ r²·exp[−F_eff(r)]
     - 本构曲线：变形前固定截去最外侧 WTOL 概率（有限参考系综）
     - 初始模量：使用完整平衡分布，不截断
   · Gibbs（第 17 页）模型共享 p0，仅替换变形响应：
     r(f) = x(f)·Lc(f)，Lc(f) = N·ξf·[(μ+1)/2 + (μ−1)/2·tanh(k1(f−k2))]

 可视化请使用同目录下的独立脚本：
   3_chain_constitutive_curves.py / 3_chain_modulus_G0_xi_f.py / 3_chain_reference_distribution.py
================================================================================
"""

import numpy as np
from scipy.special import logsumexp
from scipy.integrate import cumulative_trapezoid
from functools import lru_cache

# ===================== 全局数值兼容 =====================
# NumPy 2.x 中 np.trapz 已移除，统一使用 np.trapezoid
trapz_func = np.trapezoid

# ============ 核心物理参数 ============
mu       = 7.6      # 展开态/折叠态轮廓长度比
Delta_E  = 11.9     # 内部态能量尺度（kBT）
xi_f     = 3.6      # 折叠态单域长度（lp）
delta    = 0.05     # 支撑区间保护
DR       = 0.005    # 均匀网格间距（lp）

# 有限参考系综：在变形前固定去除最外侧 WTOL 概率并重新归一化。
# 此近似使保留链都满足不可伸长约束；不能解释为完整平衡系综的极限。
WTOL = 5e-3
CONTOUR_MARGIN = 0.995
N_QUAD = 512

# ============ Gibbs（第 17 页）唯象参数 ============
K1 = 6.5
K2 = 1.48


# ===================== 基础物理函数 =====================
def phi(x):
    """WLC 自由能的无量纲部分。"""
    x = np.clip(x, 0, 1 - 1e-12)
    return x**2 * (3 - 2*x) / (4 * (1 - x))


def U_of_n(n):
    """内部变量 n 的势能项。"""
    return Delta_E * n - Delta_E * np.cos(2 * np.pi * n)


def fMS(x):
    """Marko–Siggia 无量纲链力。"""
    x = np.clip(x, 0.0, 1.0 - 1e-9)
    return 0.25 / (1.0 - x)**2 - 0.25 + x


def fMSp(x):
    """df_MS/dx。"""
    x = np.clip(x, 0.0, 1.0 - 1e-9)
    return 0.5 / (1.0 - x)**3 + 1.0


# ===================== 单域退火统计 =====================
def domain_statistics(r, M=N_QUAD):
    """直接积分内部态：返回 log Z、有效力和有效刚度，不封顶或裁掉负刚度。

    f_eff = <f_n>，f_eff' = <f_n'> - Var(f_n)，kBT = lp = 1。
    每个端距将 n 的积分区间限定为 Lc(n) > r。
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0) or np.any(r >= mu * xi_f):
        raise ValueError('Protein end distance must satisfy 0 <= r < mu*xi_f')
    q, w = np.polynomial.legendre.leggauss(M)
    lower = np.maximum(0.0, (r / xi_f - 1.0) / (mu - 1.0))
    half = 0.5 * (1.0 - lower)
    n = lower[..., None] + half[..., None] * (q + 1.0)
    Lc = xi_f * (1.0 + (mu - 1.0) * n)
    x = r[..., None] / Lc
    lw = -Lc * phi(x) - U_of_n(n) + np.log(half[..., None] * w)
    logZ = logsumexp(lw, axis=-1)
    prob = np.exp(lw - logZ[..., None])
    force = fMS(x)
    f = np.sum(prob * force, axis=-1)
    fp = np.sum(prob * fMSp(x) / Lc, axis=-1) - np.sum(
        prob * (force - f[..., None])**2, axis=-1)
    return logZ, f, fp


def single_domain_u(M=N_QUAD):
    """在均匀网格上计算单域未归一化径向权重 u(r) ∝ exp(−F_eff(r))。"""
    r = np.arange(0.0, mu * xi_f - delta, DR)
    logZ, _, _ = domain_statistics(r, M)
    return r, np.exp(logZ - logZ.max())


def finite_reference(r, p0, tail=WTOL):
    """变形前固定参考分布；截去 tail 尾概率，绝不随 lambda 删链。"""
    cdf = cumulative_trapezoid(p0, r, initial=0.0)
    cdf /= cdf[-1]
    cutoff = np.interp(1.0 - tail, cdf, r)
    keep = r < cutoff
    rg = np.append(r[keep], cutoff)
    pg = np.interp(rg, r, p0)
    pg /= trapz_func(pg, rg)
    return rg, pg


# ===================== 网络模型 =====================
class ChainN1:
    """退火蛋白网络（N=1 单域）：参考态 p0 ∝ r²·u(r)，截断后固定。"""

    def __init__(self, r, u, tail=WTOL):
        p0 = r**2 * u
        self.r, self.p0 = finite_reference(r, p0, tail)
        self.L = mu * xi_f
        self.f, self.fp = domain_statistics(self.r)[1:]
        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

    def sigma(self, lam):
        out = np.zeros_like(lam, dtype=float)
        for i, lm in enumerate(lam):
            f1 = domain_statistics(lm * self.r)[1]
            f2 = domain_statistics(lm**-0.5 * self.r)[1]
            out[i] = trapz_func(self.p0 * self.r *
                               (f1 - lm**-1.5 * f2), self.r) / 3.0
        return out

    def E0(self):
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r)


class WLCNetwork:
    """固定轮廓 WLC；解折叠基线继承蛋白网络的固定参考分布。"""

    def __init__(self, L, delta=delta, dr=DR, reference=None, tail=WTOL):
        self.L = float(L)
        self._own_ref = reference is None
        if reference is None:
            r = np.arange(0.0, self.L - delta, dr)
            p0 = r**2 * np.exp(-self.L * phi(r / self.L))
            self.r, self.p0 = finite_reference(r, p0, tail)
        else:
            self.r, self.p0 = (np.asarray(v, dtype=float) for v in reference)
        self.f = self.force(self.r)
        self.fp = fMSp(self.r / self.L) / self.L
        self.R0 = np.sqrt(trapz_func(self.r**2 * self.p0, self.r))

    def force(self, r):
        if np.any(np.asarray(r) >= self.L):
            raise ValueError('WLC end distance must remain below its contour length')
        return fMS(np.asarray(r) / self.L)

    def sigma(self, lam):
        return np.array([trapz_func(self.p0 * self.r *
                         (self.force(lm * self.r) - lm**-1.5 *
                          self.force(lm**-0.5 * self.r)), self.r) / 3.0
                         for lm in lam])

    def E0(self):
        return 0.5 * trapz_func(
            (self.r**2 * self.fp + self.r * self.f) * self.p0, self.r)


class GibbsNetwork:
    """参考态继承原蛋白网络，仅将变形自由能换为积分 f_G 得到的 F_G。"""

    def __init__(self, reference, N=1):
        self.r, self.p0 = reference
        self.N = N
        self.L = N * mu * xi_f

    def contour(self, f):
        t = np.tanh(K1 * (f - K2))
        L = self.N * xi_f * ((mu + 1) / 2 + (mu - 1) * t / 2)
        Lp = self.N * xi_f * (mu - 1) * K1 * (1 - t*t) / 2
        return L, Lp

    def force_stiffness(self, r):
        r = np.asarray(r, dtype=float)
        if np.any(r < 0) or np.any(r >= self.L):
            raise ValueError('gibbs end distance must satisfy 0 <= r < Lmax')
        lo = np.zeros_like(r)
        hi = np.full_like(r, 1.0 - 1e-10)
        for _ in range(60):
            x = (lo + hi) / 2
            distance = x * self.contour(fMS(x))[0]
            below = distance < r
            lo = np.where(below, x, lo)
            hi = np.where(below, hi, x)
        x = (lo + hi) / 2
        f = fMS(x)
        L, Lp = self.contour(f)
        fp = fMSp(x) / (L + x * Lp * fMSp(x))
        return f, fp

    def sigma(self, lam):
        return np.array([trapz_func(self.p0 * self.r *
            (self.force_stiffness(lm * self.r)[0] - lm**-1.5 *
             self.force_stiffness(lm**-0.5 * self.r)[0]), self.r) / 3
             for lm in lam])

    def E0(self):
        f, fp = self.force_stiffness(self.r)
        return trapz_func(self.p0 * (self.r**2 * fp + self.r * f), self.r) / 2

    def chain_energy(self, r, order=512):
        """F_G(r)=integral_0^r f_G(s)ds，独立于网络应力积分。"""
        r = np.asarray(r)
        q, w = np.polynomial.legendre.leggauss(order)
        t = (q + 1) / 2
        return r * np.sum(self.force_stiffness(r[..., None] * t)[0] * w / 2,
                          axis=-1)

    def network_energy(self, lam, order=512):
        return trapz_func(self.p0 * (self.chain_energy(lam * self.r, order) +
            2 * self.chain_energy(lam**-0.5 * self.r, order)), self.r) / 3


def lam_upper_bound(net, Lwall):
    """所有保留参考链均在轮廓以内；相同参考态及轮廓有相同上限。"""
    return CONTOUR_MARGIN * Lwall / net.r[-1]


def compute_constitutive(N=1, n_lam=160):
    """计算四模型本构曲线，返回绘图与落盘所需的全部数据。

    返回 dict 键：lam, sig（protein）、lam_f, sig_f（WLC folded）、
    lam_u, sig_u（WLC unfolded）、sig_g（Gibbs）、chain, wlc_folded,
    wlc_unfolded, gibbs 四个网络对象，以及 lam_max。
    """
    r, u = single_domain_u()
    ch = ChainN1(r, u)

    hi = lam_upper_bound(ch, ch.L)
    lam = np.linspace(1.0, hi, n_lam)
    sig = ch.sigma(lam)

    wlc_folded = WLCNetwork(xi_f)
    wlc_unfolded = WLCNetwork(mu * xi_f, reference=(ch.r, ch.p0))
    gibbs_net = GibbsNetwork((ch.r, ch.p0), N=N)

    lam_f = np.linspace(1.0, lam_upper_bound(wlc_folded, wlc_folded.L), n_lam)
    sig_f = wlc_folded.sigma(lam_f)
    lam_u = lam  # 同一参考分布、同一最大轮廓长度、同一有效伸长区间
    sig_u = wlc_unfolded.sigma(lam_u)
    sig_g = gibbs_net.sigma(lam)

    return dict(lam=lam, sig=sig, lam_f=lam_f, sig_f=sig_f,
                lam_u=lam_u, sig_u=sig_u, sig_g=sig_g,
                chain=ch, wlc_folded=wlc_folded,
                wlc_unfolded=wlc_unfolded, gibbs=gibbs_net,
                lam_max=hi)


def validate_gibbs(net):
    """验证反演、初始模量以及自由能导数；返回验证文本行。"""
    distances = np.linspace(0, 0.995 * net.L, 300)
    force, _ = net.force_stiffness(distances)
    # 从 f 独立反演 Marko-Siggia x，检查 r(f)。
    lo, hi = np.zeros_like(force), np.ones_like(force)
    for _ in range(60):
        x = (lo + hi) / 2
        below = fMS(x) < force
        lo = np.where(below, x, lo)
        hi = np.where(below, hi, x)
    residual = np.max(np.abs((lo + hi) / 2 * net.contour(force)[0] - distances))
    assert residual < 1e-10
    h = 1e-4
    slope = (net.sigma([1+h])[0] - net.sigma([1-h])[0]) / (2*h)
    assert np.isclose(slope, net.E0(), rtol=1e-6)
    assert abs(net.sigma([1.0])[0]) < 1e-12
    lines = [f'force inversion max residual: {residual:.8g}',
             f'E0 analytic: {net.E0():.10g}; finite difference: {slope:.10g}']
    for lm in [2.0, 6.0]:
        derivative = (net.network_energy(lm+h) - net.network_energy(lm-h)) / (2*h)
        stress = net.sigma([lm])[0]
        energy256 = net.network_energy(lm, 256)
        energy512 = net.network_energy(lm, 512)
        assert np.isclose(derivative, stress, rtol=2e-4, atol=1e-7)
        assert np.isclose(energy256, energy512, rtol=2e-5, atol=1e-7)
        lines.append(f'lambda={lm:g}: dF/dlambda={derivative:.10g}, '
                     f'sigma={stress:.10g}, F(256)={energy256:.10g}, '
                     f'F(512)={energy512:.10g}')
    return lines


# ===================== 模量计算（完整平衡参考态，不截断） =====================
@lru_cache(maxsize=None)
def gauss_unit(points):
    """[0,1] 区间 Gauss-Legendre 节点与权重（带缓存）。"""
    q, w = np.polynomial.legendre.leggauss(points)
    return (q + 1) / 2, w / 2


def Lc_of_n(n, N, xi_f):
    """轮廓长度 Lc 必须严格依赖状态变量 n，而非力 f。"""
    return N * xi_f * (1 + n*(mu - 1))


def annealed_statistics(r, N, xi_f, n_points=512):
    """logZ、平均力、退火刚度、固定态平均刚度及条件力方差。"""
    r = np.asarray(r)
    lower = np.maximum(0., (r / (N * xi_f) - 1) / (mu - 1))
    if np.any(lower >= 1):
        raise ValueError('End distance must remain below maximum contour length')
    t, wn = gauss_unit(n_points)
    n = lower[..., None] + (1 - lower[..., None]) * t
    Lc = Lc_of_n(n, N, xi_f)
    x = r[..., None] / Lc
    lw = -Lc * phi(x) - U_of_n(n) + np.log((1 - lower[..., None]) * wn)
    logZ = logsumexp(lw, axis=-1)
    conditional = np.exp(lw - logZ[..., None])
    fn = fMS(x)
    f = np.sum(conditional * fn, axis=-1)
    frozen_fp = np.sum(conditional * fMSp(x) / Lc, axis=-1)
    variance = np.sum(conditional * (fn - f[..., None])**2, axis=-1)
    return logZ, f, frozen_fp - variance, frozen_fp, variance


def build_annealed(N, xi_f, n_points=512, r_points=1024):
    """完整平衡参考态：r = Lmax·t² 网格保留全部支撑并在小端距加密。

    mass ∝ r²·exp(−F_eff)·wr 即完整平衡径向分布，不做尾部截断。
    """
    t, wt = gauss_unit(r_points)
    Lmax = N * xi_f * mu
    r = Lmax * t**2
    wr = wt * 2 * Lmax * t
    logZ, f, fp, frozen_fp, variance = annealed_statistics(r, N, xi_f, n_points)
    logmass = logZ + 2 * np.log(r) + np.log(wr)
    mass = np.exp(logmass - logsumexp(logmass))
    return dict(r=r, mass=mass, f=f, fp=fp, frozen_fp=frozen_fp,
                variance=variance, N=N, xi_f=xi_f, n_points=n_points)


def modulus_components(pre):
    """返回（退火模量 G0、冻结模量、方差修正项），单位 ρkBT。"""
    r, w, f = pre['r'], pre['mass'], pre['f']
    frozen = 0.5 * np.sum(w * (r*r*pre['frozen_fp'] + r*f))
    correction = 0.5 * np.sum(w * r*r*pre['variance'])
    return frozen - correction, frozen, correction


def G0_3chain_quad(N, xi_f, n_points=512, r_points=1024):
    """退火有效自由能的初始拉伸模量（保持原函数接口）。"""
    return modulus_components(build_annealed(N, xi_f, n_points, r_points))[0]


def G0_3chain_WLC(N, xi_f, points=1024):
    """固定轮廓 WLC 网络的初始模量，L = N·ξf。"""
    L = N * xi_f
    x, wx = gauss_unit(points)
    logmass = 2*np.log(x) - L*phi(x) + np.log(wx)
    mass = np.exp(logmass - logsumexp(logmass))
    return 0.5 * L * np.sum(mass * x * (x*fMSp(x) + fMS(x)))


def gibbs_force_stiffness(r, N, xi_f):
    """返回 f_G 及 df_G/dr；推导见 derivations/gibbs_initial_modulus.md。"""
    r = np.asarray(r, dtype=float)
    if np.any(r < 0) or np.any(r >= N*mu*xi_f):
        raise ValueError('gibbs distance must be below maximum contour length')
    lo, hi = np.zeros_like(r), np.ones_like(r)
    for _ in range(60):
        x = (lo + hi) / 2
        f = fMS(x)
        t = np.tanh(K1*(f-K2))
        L = N*xi_f*((mu+1)/2 + (mu-1)*t/2)
        below = x*L < r
        lo, hi = np.where(below, x, lo), np.where(below, hi, x)
    x = (lo + hi) / 2
    f = fMS(x)
    t = np.tanh(K1*(f-K2))
    L = N*xi_f*((mu+1)/2 + (mu-1)*t/2)
    Lp = N*xi_f*(mu-1)*K1*(1-t*t)/2
    fp = fMSp(x)/(L + x*Lp*fMSp(x))
    return f, fp


def G0_3chain_gibbs(pre):
    """Gibbs 模型初始模量，参考态与退火网络相同。"""
    r, w = pre['r'], pre['mass']
    f, fp = gibbs_force_stiffness(r, pre['N'], pre['xi_f'])
    return 0.5*np.sum(w*(r*r*fp+r*f))


def calculate_pn(N, xi_f, n_points=512, r_points=1024):
    """内部态边缘分布 p(n) ∝ wn·Z_n，Z_n 对每个 n 完整积分 r²exp(−F_d)dr。"""
    n, wn = gauss_unit(n_points)
    t, wt = gauss_unit(r_points)
    L = Lc_of_n(n, N, xi_f)
    # 对每个 n 完整积分 r² exp[-F_d] dr。
    lw = (3*np.log(L[:, None]) + 2*np.log(t)[None, :] -
          L[:, None]*phi(t)[None, :] - U_of_n(n)[:, None] + np.log(wt)[None, :])
    logzn = logsumexp(lw, axis=1)
    pn = np.exp(logzn - logsumexp(logzn + np.log(wn)))
    return n, pn


def validate_modulus():
    """退火模量四重校验：细化网格、分部积分恒等式、应力差分、修正非负。"""
    lines = []
    for xi in [1., 3.6, 10., 100.]:
        pre = build_annealed(1, xi)
        g, frozen, correction = modulus_components(pre)
        finer = G0_3chain_quad(1, xi, n_points=1024, r_points=2048)
        assert np.isclose(g, finer, rtol=1e-6, atol=1e-7), (xi,g,finer)
        # 平衡径向分布的分部积分恒等式，独立核对刚度积分。
        rf = pre['r'] * pre['f']
        identity = 0.5 * (np.sum(pre['mass'] * rf**2) - 9)
        assert np.isclose(np.sum(pre['mass'] * rf), 3, rtol=1e-6)
        assert np.isclose(g, identity, rtol=1e-6, atol=1e-7)
        # 可忽略尾部仅用于有限差分，生产模量使用完整积分。
        keep = pre['mass'] > 1e-22
        r, w = pre['r'][keep], pre['mass'][keep]
        def stress(lam):
            f1 = annealed_statistics(lam*r, 1, xi)[1]
            f2 = annealed_statistics(lam**-0.5*r, 1, xi)[1]
            return np.sum(w*r*(f1-lam**-1.5*f2))/3
        h=1e-5
        slope=(stress(1+h)-stress(1-h))/(2*h)
        assert np.isclose(g,slope,rtol=1e-6,atol=1e-7)
        assert correction >= 0 and g <= frozen + 1e-12
        line=(f'xi_f={xi:g}: G0={g:.10g}, frozen={frozen:.10g}, '
              f'variance_correction={correction:.10g}, refined={finer:.10g}, '
              f'sigma_derivative={slope:.10g}, identity={identity:.10g}')
        print(line); lines.append(line)
    return lines


def validate_gibbs_modulus():
    """Gibbs 模量校验：细化网格、应力差分、单链刚度中心差分。"""
    lines = []
    for xi in [1.,3.6,10.,100.]:
        pre = build_annealed(1,xi)
        g = G0_3chain_gibbs(pre)
        fine = G0_3chain_gibbs(build_annealed(1,xi,n_points=1024,r_points=2048))
        assert np.isclose(g,fine,rtol=1e-6,atol=1e-7), (xi,g,fine)
        keep = pre['mass'] > 1e-22
        r,w = pre['r'][keep], pre['mass'][keep]
        def stress(lam):
            f1 = gibbs_force_stiffness(lam*r,1,xi)[0]
            f2 = gibbs_force_stiffness(lam**-0.5*r,1,xi)[0]
            return np.sum(w*r*(f1-lam**-1.5*f2))/3
        h=1e-5
        slope=(stress(1+h)-stress(1-h))/(2*h)
        assert stress(1.) == 0.
        assert np.isclose(g,slope,rtol=1e-6,atol=1e-7)
        # 单链刚度独立中心差分，覆盖过渡段及高伸长段。
        rr=np.linspace(.001,.95,200)*mu*xi
        f,fp=gibbs_force_stiffness(rr,1,xi)
        dh=1e-6*xi
        fd=(gibbs_force_stiffness(rr+dh,1,xi)[0]-
            gibbs_force_stiffness(rr-dh,1,xi)[0])/(2*dh)
        assert np.allclose(fp,fd,rtol=1e-6,atol=1e-7)
        line=f'gibbs xi_f={xi:g}: G0={g:.10g}, refined={fine:.10g}, sigma_derivative={slope:.10g}'
        print(line);lines.append(line)
    return lines
