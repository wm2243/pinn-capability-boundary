# -*- coding: utf-8 -*-
"""
空间权重族（论文第三章六族）——自包含纯函数, 训练(torch)与冻结谱/分析(numpy)共用
同一条映射, 避免两套实现漂移。

统一对比度: t = |r| / (bg + eps) >= 0（残差幅值对比度, 无上界）。
  q = t/(1+t) in [0,1)  软饱和
六族（beta>1, 唯一变量是对比度的方向与形状）:
  rational  w = 1+(beta-1) q = (1+beta t)/(1+t)  in [1,beta]   增权（主方法）
  linear    w = 1+(beta-1) clip(t,0,1)           in [1,beta]   增权·线性截断
  band      w = 1+(beta-1) 1{q > band_q}         in {1,beta}   增权·激波带硬聚焦
  down_lin  w = 1-(1-1/beta) q                   in [1/beta,1] 减权·线性
  down_inv  w = 1/[1+(beta-1) q]                 in [1/beta,1] 减权·倒数
  uniform   w = 1
理论常数（加权范数双向界, 确定性）:
  ||e|| <= C_pde/sqrt(w_min) ||r||_{L2(w)}; 增权 w_min=1 稳定常数不变, 减权 w_min=1/beta 放大 sqrt(beta)。
"""
import numpy as np

FAMILIES = ("rational", "linear", "band", "down_lin", "down_inv", "uniform")
UP_FAMILIES = ("rational", "linear", "band")
DOWN_FAMILIES = ("down_lin", "down_inv")


def family_weight_np(family, t, beta, eps=1e-12, band_q=0.5):
    """numpy 版: t 为 ndarray（或标量）, 返回同形状权重。"""
    b = float(beta)
    t = np.asarray(t, dtype=np.float64)
    q = t / (1.0 + t)
    if family == "rational":
        return 1.0 + (b - 1.0) * q
    if family == "linear":
        return 1.0 + (b - 1.0) * np.clip(t, 0.0, 1.0)
    if family == "band":
        return 1.0 + (b - 1.0) * (q > band_q).astype(np.float64)
    if family == "down_lin":
        return 1.0 - (1.0 - 1.0 / b) * q
    if family == "down_inv":
        return 1.0 / (1.0 + (b - 1.0) * q + eps)
    if family == "uniform":
        return np.ones_like(t)
    raise ValueError("unknown family %r, valid=%s" % (family, FAMILIES))


def family_weight(family, t, beta, eps=1e-12, band_q=0.5):
    """torch 训练版: t 为 Tensor; 也兼容 numpy（转发到 _np）。"""
    try:
        import torch
        if isinstance(t, torch.Tensor):
            b = float(beta)
            q = t / (1.0 + t)
            if family == "rational":
                return 1.0 + (b - 1.0) * q
            if family == "linear":
                return 1.0 + (b - 1.0) * torch.clamp(t, 0.0, 1.0)
            if family == "band":
                return 1.0 + (b - 1.0) * (q > band_q).to(t.dtype)
            if family == "down_lin":
                return 1.0 - (1.0 - 1.0 / b) * q
            if family == "down_inv":
                return 1.0 / (1.0 + (b - 1.0) * q + eps)
            if family == "uniform":
                return torch.ones_like(t)
            raise ValueError("unknown family %r" % family)
    except ImportError:
        pass
    return family_weight_np(family, t, beta, eps=eps, band_q=band_q)


def w_min_theory(family, beta):
    """对比度 t→∞(q→1) 时的权重下界: 减权族 1/beta, 其余 1。"""
    b = float(beta)
    return 1.0 / b if family in DOWN_FAMILIES else 1.0


def stability_factor(family, beta):
    """C_w/C_pde = 1/sqrt(w_min): 增权=1, 减权=sqrt(beta)。"""
    return float(1.0 / np.sqrt(max(w_min_theory(family, beta), 1e-12)))


def multiplier_phi_rational(t, beta):
    """有理增权的乘子 m_phi = 2w + 2t w'（确定性恒等式）, 理论上 m_phi in [2, 2 beta]。"""
    t = np.asarray(t, dtype=np.float64)
    b = float(beta)
    w = (1.0 + b * t) / (1.0 + t)
    wp = (b - 1.0) / (1.0 + t) ** 2
    return 2.0 * w + 2.0 * t * wp


def psi2pp_rational(t, beta):
    """有理族势函数二阶导 Psi''(t)=2(1+3 beta t+3 beta t^2+beta t^3)/(1+t)^3, 对 beta>=1,t>=0 恒正。"""
    t = np.asarray(t, dtype=np.float64)
    b = float(beta)
    return 2.0 * (1.0 + 3.0 * b * t + 3.0 * b * t ** 2 + b * t ** 3) / (1.0 + t) ** 3
