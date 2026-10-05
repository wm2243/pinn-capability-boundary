# -*- coding: utf-8 -*-
"""
有理加权 PDE 损失（V6 修订，向后兼容旧接口）

统一权重：w(t)=(1+βt)/(1+t), t=|r|/bg, bg=背景尺度(批内|r|中位数)，故 w∈[1,β]。
损失 L = mean( w * r^2 )。

—— 理论对应（务必与论文一致）——
* adaptive（时变、未冻结）：w 依赖当前 r，自动微分含 ∂w/∂r，梯度乘子为 m_φ=2w+t w'（命题3）；
* 冻结/固定权重（freeze_residual_field / warmstart / oracle）：w 不依赖当前 r，梯度乘子为 2w；
* raw 模式保持 w≥1（测度不变性成立）；selfnorm 模式 w̃=w/mean(w)（均值1，纯注意力重分配、
  不改变损失总尺度，用于消融“尺度效应 vs 重分配效应”，此时 w̃ 不逐点≥1，用范数等价版本）。

V6 新增能力：
  - set_beta()                : 训练中时变 β（β 课程，Phase 1），不重建对象
  - weight_norm='raw/selfnorm': 原始权重 / 自归一化(均值1)
  - 背景尺度 EMA（bg_ema_decay）: 平滑批内中位数，抑制背景尺度抖动
  - 空间权重场 EMA（spatial_ema_decay + 固定 ref grid）: 得到“缓变 W(x)”，贴合理论缓变假设、抑锯齿
  - freeze_residual_field()  : 在当前 θ 冻结“残差对比度场 t(x)”，之后只随 β 现算 w ——
                               这是 Phase 1 冻结 W、以及 C1“同一邻域扫描 β 看静态 κ(β)”的正确实现；
                               冻结的是 t(x)，改 β 无需重算残差。
模式：adaptive / oracle_spatial(真解梯度上界) / warmstart_spatial(粗解冻结，可实现)。
"""

import torch
import torch.nn as nn
import numpy as np


class RationalWeightingLoss(nn.Module):
    VALID_MODES = ("adaptive", "oracle_spatial", "warmstart_spatial", "spatial")
    VALID_NORM = ("raw", "selfnorm")

    def __init__(self, config, device):
        super().__init__()
        self.beta = float(config.get("loss_beta", 1.0))
        self.nu = float(config.get("nu", 0.01))
        mode = config.get("weight_mode", "adaptive")
        self.weight_mode = "oracle_spatial" if mode == "spatial" else mode
        if self.weight_mode not in self.VALID_MODES:
            raise ValueError(
                f"weight_mode 必须是标量字符串 {self.VALID_MODES}，得到 {mode!r}（类型 {type(mode).__name__}）；"
                f"扫描脚本必须在每个实验里用 config['weight_mode']=mode(标量) 覆盖，不能传列表。")
        self.device = device
        self.eps = 1e-12

        # 归一化 / 缓变选项（默认与旧版一致：raw、无 EMA）
        self.weight_norm = config.get("weight_norm", "raw")
        if self.weight_norm not in self.VALID_NORM:
            raise ValueError(f"weight_norm ∈ {self.VALID_NORM}，得到 {self.weight_norm}")
        self.bg_ema_decay = float(config.get("bg_ema_decay", 0.0))          # 背景中位数 EMA，0=关闭
        self.spatial_ema_decay = float(config.get("spatial_ema_decay", 0.0))  # 空间权重场 EMA，0=关闭
        self._bg_ema = None                                                # 背景尺度 EMA（标量）

        # 固定参考网格（空间缓变权重场 / 冻结残差对比度场共用）
        n_ref = int(config.get("weight_ref_grid", 2048))
        self.register_buffer("_ref_grid", torch.linspace(-1, 1, n_ref, device=device).view(-1))
        self.register_buffer("_ema_t_field", None)   # ref grid 上的残差对比度 t(x) EMA
        self.register_buffer("_frozen_t_field", None)  # Phase1 冻结的 t(x)
        self.field_frozen = False

        # warmstart 冻结权重（旧逻辑保留）
        self.register_buffer("_ws_grid", None)
        self.register_buffer("_ws_w", None)
        self.warmstart_frozen = False

    # ---------------- 时变 β ----------------
    def set_beta(self, beta):
        self.beta = float(beta)

    # ---------------- 统一权重公式 ----------------
    def _w(self, t):
        return (1.0 + self.beta * t) / (1.0 + t + self.eps)

    def _bg_scale(self, a_abs):
        """背景尺度=批内|r|中位数，可选 EMA 平滑。全常数/趋零返回 None（权重恒1）。"""
        bg = torch.median(a_abs)
        if not torch.isfinite(bg) or bg < 1e-12:
            return None
        bg = float(bg)
        if self.bg_ema_decay > 0:
            if self._bg_ema is None:
                self._bg_ema = bg
            else:
                d = self.bg_ema_decay
                self._bg_ema = d * self._bg_ema + (1 - d) * bg
            bg = self._bg_ema
        return max(bg, 1e-12)

    # ---------------- oracle：真解解析梯度 ----------------
    def _oracle_t(self, x):
        return (1.0 / (2.0 * self.nu)) * (1.0 / torch.cosh(x.reshape(-1) / (2.0 * self.nu))) ** 2

    # ---------------- 空间场插值工具 ----------------
    def _interp_field(self, field, x):
        x = x.reshape(-1)
        idx = torch.searchsorted(self._ref_grid, x).clamp(1, len(self._ref_grid) - 1)
        x0, x1 = self._ref_grid[idx - 1], self._ref_grid[idx]
        f0, f1 = field[idx - 1], field[idx]
        a = ((x - x0) / (x1 - x0 + self.eps)).clamp(0.0, 1.0)
        return f0 + a * (f1 - f0)

    def update_residual_field(self, model, pde_engine):
        """在固定 ref grid 上更新残差对比度场（空间 EMA 缓变 W 用）。
        注意：残差需二阶 autograd，本方法【不能】被 no_grad 包裹；只对存入的场 detach。"""
        g = self._ref_grid.view(-1, 1)
        r = pde_engine.compute_residual(model, g).reshape(-1)
        a = r.abs().detach()
        bg = self._bg_scale(a)
        t = torch.zeros_like(a) if bg is None else a / bg
        if self.spatial_ema_decay > 0 and self._ema_t_field is not None:
            d = self.spatial_ema_decay
            self._ema_t_field = d * self._ema_t_field + (1 - d) * t
        else:
            self._ema_t_field = t.contiguous()

    def freeze_residual_field(self, model=None, pde_engine=None):
        """Phase 1 / C1：冻结当前残差对比度场 t(x)。之后改 β 只重算 w、不重算残差。
        注意：残差需二阶 autograd，本方法【不能】被 no_grad 包裹；只对存入的场 detach。"""
        if self._ema_t_field is not None:
            frozen = self._ema_t_field.detach().clone()
        else:
            if model is None or pde_engine is None:
                raise RuntimeError("冻结残差场需传入 model 与 pde_engine，或先 update_residual_field")
            g = self._ref_grid.view(-1, 1)
            with torch.enable_grad():  # 残差含 u_x/u_xx 需建图，即便外部处于 no_grad 也强制开启
                r = pde_engine.compute_residual(model, g).reshape(-1)
            r = r.abs().detach()
            bg = self._bg_scale(r)
            frozen = torch.zeros_like(r) if bg is None else (r / bg).contiguous()
        self._frozen_t_field = frozen
        self.field_frozen = True
        with torch.no_grad():  # 纯权重统计，无需建图
            return float(self._w(frozen).min()), float(self._w(frozen).max())

    def unfreeze_residual_field(self):
        self.field_frozen = False
        self._frozen_t_field = None

    @torch.no_grad()
    def load_frozen_field(self, t_field, beta=None):
        """直接灌入外部冻结的残差对比度场（受控 β 扫描共享同一好盆地时用）。"""
        if beta is not None:
            self.beta = float(beta)
        self._frozen_t_field = t_field.to(self.device).reshape(-1).contiguous()
        self.field_frozen = True

    # ---------------- warmstart（旧逻辑，网格加密到与 ν 匹配）----------------
    @torch.no_grad()
    def freeze_warmstart(self, model, grid_x=None):
        if grid_x is None:
            # 修复旧版 2048 点在小 ν 时对激波欠采样：至少保证激波层内 ~200 点
            n_grid = int(max(2048, np.ceil(400.0 / max(self.nu, 1e-4))))
            grid_x = torch.linspace(-1, 1, min(n_grid, 200000), device=self.device).view(-1, 1)
        grid_x = grid_x.to(self.device).view(-1)
        model.eval()
        u = model(grid_x.view(-1, 1)).reshape(-1)
        du = torch.zeros_like(u)
        du[1:-1] = (u[2:] - u[:-2]) / (grid_x[2:] - grid_x[:-2] + self.eps)
        du[0] = (u[1] - u[0]) / (grid_x[1] - grid_x[0] + self.eps)
        du[-1] = (u[-1] - u[-2]) / (grid_x[-1] - grid_x[-2] + self.eps)
        bg = self._bg_scale(du.abs())
        t = torch.zeros_like(du) if bg is None else du.abs() / bg
        w = self._w(t)
        order = torch.argsort(grid_x)
        self._ws_grid = grid_x[order].contiguous()
        self._ws_w = w[order].contiguous()
        self.warmstart_frozen = True
        model.train()
        return float(w.min()), float(w.max())

    def _interp_warmstart(self, x):
        x = x.reshape(-1)
        idx = torch.searchsorted(self._ws_grid, x).clamp(1, len(self._ws_grid) - 1)
        x0, x1 = self._ws_grid[idx - 1], self._ws_grid[idx]
        w0, w1 = self._ws_w[idx - 1], self._ws_w[idx]
        a = ((x - x0) / (x1 - x0 + self.eps)).clamp(0.0, 1.0)
        return w0 + a * (w1 - w0)

    # ---------------- 对外：逐点权重 ----------------
    def get_current_weights(self, residual, x_pde=None):
        residual = residual.reshape(-1)

        if self.weight_mode == "adaptive":
            if self.field_frozen and self._frozen_t_field is not None:
                # Phase1/C1：冻结残差对比度场，权重只随当前 β 变
                if x_pde is None:
                    raise RuntimeError("冻结场模式必须传入 x_pde 做插值")
                t = self._interp_field(self._frozen_t_field, x_pde)
                return self._normalize(self._w(t))
            if self.spatial_ema_decay > 0 and self._ema_t_field is not None:
                # 缓变空间权重场
                t = self._interp_field(self._ema_t_field, x_pde)
                return self._normalize(self._w(t))
            t_a = self._relative_to_bg(residual.abs())
            return self._normalize(self._w(t_a))

        if self.weight_mode == "oracle_spatial":
            if x_pde is None:
                raise RuntimeError("oracle_spatial 必须传入 x_pde")
            return self._normalize(self._w(self._oracle_t(x_pde)))

        if self.weight_mode == "warmstart_spatial":
            if not self.warmstart_frozen:
                return torch.ones_like(residual)
            if x_pde is None:
                raise RuntimeError("warmstart_spatial 冻结后必须传入 x_pde")
            return self._normalize(self._interp_warmstart(x_pde))

        return torch.ones_like(residual)

    @staticmethod
    def _relative_to_bg(a):
        bg = torch.median(a)
        if not torch.isfinite(bg) or bg < 1e-12:
            return torch.zeros_like(a)
        return a / bg

    def _normalize(self, w):
        """raw：保持 w∈[1,β]；selfnorm：除以批内均值(detach)使均值=1，纯重分配。"""
        if self.weight_norm == "selfnorm":
            return w / w.mean().detach().clamp_min(self.eps)
        return w

    # ---------------- 损失 ----------------
    def forward(self, residual, x_pde=None):
        weights = self.get_current_weights(residual, x_pde=x_pde)
        weighted_residual_sq = weights * (residual.reshape(-1) ** 2)
        loss = weighted_residual_sq.mean()
        with torch.no_grad():
            loss_dict = {
                "pde_loss": loss.item(),
                "weight_mean": float(weights.mean()),
                "weight_max": float(weights.max()),
                "weight_min": float(weights.min()),
                "beta": self.beta,
            }
        return loss, loss_dict
