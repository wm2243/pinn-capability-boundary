# -*- coding: utf-8 -*-
"""
一维稳态反应-扩散（Allen-Cahn 型）奇摄动【内部尖峰】方程：S2e 算例
================================================================================
服务理论（不以刷低 L2 为目标）：在方程类别（反应-扩散）、奇性形态（非单调钟形峰）、
盆地结构（存在竞争平凡解 u≡0）都不同的问题上，复现
  (I) 盆地选择由 σ 表示课程 + multistart 决定（减权不能把塌零解拉成带峰解）；
  (II) 方向-尺度 no-go（尖峰 Gate 点冻结截面，对角加权不改谱方向）；
  (III) 好盆地内 β 的梯度能量再分配（峰区对比度占比随 β 单调）。

控制方程：
    -eps^2 u''(x) + u(x) - u(x)^3 = 0 ,  x in (-1,1)，
    u(-1)=sqrt2 sech((-1-x0)/eps), u(1)=sqrt2 sech((1-x0)/eps)（x0=0 时为同一指数小尾值）。
同宿（homoclinic）尖峰精确解：
    u*(x) = sqrt2 sech((x-x0)/eps)，峰幅 sqrt2、宽度 O(eps)（10–90 全宽 ≈ 5.05 eps）。
代入恒等式 sech''=sech-2 sech^3 可验证其满足方程；eps 小时边界尾值下溢为 0，
故 u≡0 也是满足（近似齐次）边界的精确解——与带峰解竞争的“平凡盆地”，正是盆地选择的检验对象。
注意 -u* 亦为精确解（符号/对称破缺简并），统计盆地时须区分正峰、负峰。

接口与 ConvDiffPDE1D / TurningPointPDE1D 对齐。
"""
import numpy as np
import torch


class SpikePDE1D:
    def __init__(self, config):
        self.eps = float(config["spike_eps"])
        self.x0 = float(config.get("spike_x0", 0.0))
        self.amp = float(np.sqrt(2.0))
        assert self.eps > 0, "spike_eps 必须为正"

    def exact_np(self, x):
        return self.amp / np.cosh((x - self.x0) / self.eps)

    def bc_values(self):
        """两端精确边界值（指数小尾，eps 小时下溢为 0）。"""
        bl = self.amp / np.cosh((-1.0 - self.x0) / self.eps)
        br = self.amp / np.cosh((1.0 - self.x0) / self.eps)
        return float(bl), float(br)

    def compute_residual(self, model, x_pde):
        """残差 r = -eps^2 u_xx + u - u^3（反应-扩散，右端源项为 0）。"""
        x_pde = x_pde.detach().clone().requires_grad_(True)
        u = model(x_pde)
        u_x = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        return -(self.eps ** 2) * u_xx + u - u ** 3
