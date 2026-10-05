# -*- coding: utf-8 -*-
"""
权函数族（V9 实验 E3；E1/E2 也统一走这里，family='rational' 与主方法逐点恒等）
================================================================================
统一对比度：t=|r|/bg（与 RationalWeightingLoss 同一口径，无上界），再映射到
  q_soft = t/(1+t) ∈ [0,1)   （软饱和，有理权在 q_soft 下恰为线性）
  q_hard = clip(t,0,1)       （硬截断，对照“线性不饱和尚形”）

六族（contrast=β>1，唯一变量是 contrast 的方向与形状）：
  rational  w=1+(β-1) q_soft = (1+β t)/(1+t)  ∈[1,β]   增权（==主方法，逐点恒等）
  linear    w=1+(β-1) q_hard                   ∈[1,β]   增权·线性截断（对照饱和形状）
  band      w=1+(β-1) 1{q_soft>q0}             ∈{1,β}   增权·激波带硬聚焦
  down_lin  w=1-(1-1/β) q_soft                 ∈[1/β,1] 减权·线性
  down_inv  w=1/[1+(β-1) q_soft]               ∈[1/β,1] 减权·倒数（倍率与增权严格对称）
  uniform   w≡1                                         无加权基线

理论标注（与论文“加权范数双向界”一致，确定性、不依赖 SA）：
  ‖e‖_X ≤ C_pde / sqrt(w_min) · ‖r‖_{L2(w)}；
  增权族 w_min=1，稳定性常数保持 C_pde；减权族 w_min=1/β，常数放大 sqrt(β) 倍。
  stability_factor() 即 1/sqrt(w_min)，直接写进结果 CSV，供“双分离”定量对照。

shuffled=True：在【每个 batch 内】对权重做固定随机置换（保持直方图、破坏空间对应），
  训练侧复现 D2；冻结谱侧的置换在 E3 脚本里用 w_override 单独做，不经过这里。
================================================================================
"""
import torch
from losses.rational_weighting_loss1 import RationalWeightingLoss

FAMILIES = ("rational", "linear", "band", "down_lin", "down_inv", "uniform")
UP_FAMILIES = ("rational", "linear", "band")
DOWN_FAMILIES = ("down_lin", "down_inv")


def family_weight(family, t, beta, eps=1e-12, band_q=0.5):
    """纯函数：给定对比度场 t(tensor) 与 contrast=β，返回权重 w（供冻结谱 w_override 共用，
    保证训练与谱测量用的是同一条映射，避免两套实现漂移）。"""
    b = float(beta)
    q_soft = t / (1.0 + t)
    if family == "rational":
        return 1.0 + (b - 1.0) * q_soft
    if family == "linear":
        return 1.0 + (b - 1.0) * t.clamp(0.0, 1.0)
    if family == "band":
        return 1.0 + (b - 1.0) * (q_soft > band_q).to(t.dtype)
    if family == "down_lin":
        return 1.0 - (1.0 - 1.0 / b) * q_soft
    if family == "down_inv":
        return 1.0 / (1.0 + (b - 1.0) * q_soft + eps)
    if family == "uniform":
        return torch.ones_like(t)
    raise ValueError(f"unknown family {family!r}, valid={FAMILIES}")


def w_min_theory(family, beta):
    """对比度 t→∞（q→1）时的理论最小/最大权重边界。"""
    b = float(beta)
    if family in DOWN_FAMILIES:
        return 1.0 / b
    return 1.0  # 增权族 / uniform：w≥1


def stability_factor(family, beta):
    """C_w/C_pde = 1/sqrt(w_min)：增权=1，减权=sqrt(β)。"""
    return float(1.0 / (max(w_min_theory(family, beta), 1e-12) ** 0.5))


class FamilyWeightingLoss(RationalWeightingLoss):
    """只重写 _w（及 batch 内置换钩子），冻结场/插值/归一化/EMA/损失全部继承父类。"""

    VALID_FAMILY = FAMILIES

    def __init__(self, config, device, family="rational", shuffled=False, band_q=0.5):
        super().__init__(config, device)
        if family not in self.VALID_FAMILY:
            raise ValueError(f"family 必须是 {self.VALID_FAMILY}，得到 {family!r}")
        self.family = family
        self.shuffled = bool(shuffled)
        self.band_q = float(band_q)
        self._shuf_counter = 0

    def _w(self, t):
        return family_weight(self.family, t, self.beta, eps=self.eps, band_q=self.band_q)

    def get_current_weights(self, residual, x_pde=None):
        w = super().get_current_weights(residual, x_pde=x_pde)
        if self.shuffled and w.numel() > 1:
            self._shuf_counter += 1
            g = torch.Generator(device=w.device)
            g.manual_seed(100003 + self._shuf_counter)
            perm = torch.randperm(w.numel(), generator=g, device=w.device)
            w = w[ perm ]
        return w

    def w_min_theory(self):
        return w_min_theory(self.family, self.beta)

    def stability_factor(self):
        return stability_factor(self.family, self.beta)
