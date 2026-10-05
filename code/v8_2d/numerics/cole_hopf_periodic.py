# -*- coding: utf-8 -*-
"""
周期粘性 Burgers 的 Cole-Hopf 精确参考解（第二篇时变算例的金标准真值）。

方程:  u_t + u u_x = nu u_xx,  x in [-1,1]（周期 L=2）,  u(x,0) = -sin(pi x)。
Hopf-Cole 变换:  u = -2 nu d_x ln phi,  phi_t = nu phi_xx（线性热方程）。
周期初值对应的势:
    phi0(y) = exp( f(y) ),  f(y) = (1 - cos(pi y)) / (2 pi nu)。
周期化热核（Poisson 镜像, Y = y + 2m）下, 对高斯核积分有
    u(x,t) = E_w[ (x - Y)/t ],
    w ∝ exp( -(x-Y)^2/(4 nu t) + f(y) )。
小 nu 时 phi 动态范围达 exp(1/(pi nu))（nu=.01/pi 时约 e^100）, 直接 FFT 部分和会
灾难性相消; 故在 log 域用 softmax 计算权重（本实现）, 全程无相消。周期镜像 M>=1
即达机器精度。t=0 直接返回初值。

参考解精度: 生产默认 Ny=16001, nu=.01/pi、t<=1 时全域 rel-L2 约 1e-5、激波区
Linf 约 3e-5（见 verify_reference.py 的 h-细化表）; 远低于 PINN 目标误差(1e-3~1e-4)。
关键剖面可用 reference_u_quad() 的自适应积分独立复核。

经典谱/ETDRK4（spectral_burgers.py）在光滑段与大 nu 下与本解一致, 但在 nu=.01/pi
含激波段欠分辨、会欠耗散或湮灭, 仅作交叉验证, 不作真值。
"""
import numpy as np

L = 2.0
X_DOMAIN = (-1.0, 1.0)


def u_initial(x):
    """初值 u0(x) = -sin(pi x)。"""
    return -np.sin(np.pi * np.asarray(x, dtype=np.float64))


def _f0(y, nu):
    return (1.0 - np.cos(np.pi * y)) / (2.0 * np.pi * nu)


def _u_one_t(nu, t, x, Ny=16001, M=2, chunksize=256):
    """单个时刻 t>0 在任意 x 处的参考解（log 域周期镜像积分, x 分块控内存）。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.linspace(-1.0, 1.0, Ny)
    ms = np.arange(-M, M + 1)
    Y = np.concatenate([y + 2.0 * m for m in ms])            # (P,)
    f = np.concatenate([_f0(y, nu)] * len(ms))               # 周期, f(y+2m)=f(y)
    wcoef = (1.0 / t)
    out = np.empty_like(x)
    for s in range(0, x.shape[0], chunksize):
        xb = x[s:s + chunksize, None]                        # (b,1)
        Yb = Y[None, :]                                      # (1,P)
        a = -(xb - Yb) ** 2 / (4.0 * nu * t) + f[None, :]
        a -= a.max(axis=1, keepdims=True)
        p = np.exp(a)
        p /= p.sum(axis=1, keepdims=True)
        out[s:s + xb.shape[0]] = (p * (xb - Yb) * wcoef).sum(axis=1)
    return out


def reference_solution(nu, x, t, Ny=16001, M=2, chunksize=256):
    """在矩形网格 (t, x) 上生成参考解。

    参数
    ----
    nu : float, 运动粘度（Raissi 口径取 0.01/pi）。
    x  : array (Nx,), 空间评估点。
    t  : array (Nt,), 时间评估点（可含 0）。
    Ny : int, 周期势的梯形离散点数（h-细化验证精度, 默认 16001）。
    M  : int, 周期镜像数（每侧 M 个, M>=1 已机器精度）。
    返回
    ----
    U : ndarray (Nt, Nx), 参考解; 另返回 x, t 的实际数组。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    U = np.empty((t.shape[0], x.shape[0]), dtype=np.float64)
    for i, ti in enumerate(t):
        if ti <= 1e-12:
            U[i] = u_initial(x)
        else:
            U[i] = _u_one_t(nu, ti, x, Ny=Ny, M=M, chunksize=chunksize)
    return U, x, t


def reference_u_quad(nu, t, x, M=2, epsrel=1e-11, limit=400):
    """scipy 自适应高精度积分版（不依赖梯形网格）, 用于关键点独立复核。"""
    from scipy.integrate import quad
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    def fper(z):
        yp = np.mod(z + 1.0, 2.0) - 1.0
        return _f0(yp, nu)

    def one(xi):
        def g(z):
            return np.exp(-(xi - z) ** 2 / (4.0 * nu * t) + fper(z))

        def gx(z):
            return (xi - z) / t * g(z)
        phi = num = 0.0
        for m in range(-M, M + 1):
            a, b = -1.0 + 2.0 * m, 1.0 + 2.0 * m
            phi += quad(g, a, b, epsabs=0.0, epsrel=epsrel, limit=limit)[0]
            num += quad(gx, a, b, epsabs=0.0, epsrel=epsrel, limit=limit)[0]
        return num / phi
    return np.array([one(xi) for xi in x])
