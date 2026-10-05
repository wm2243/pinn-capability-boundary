# -*- coding: utf-8 -*-
"""
稳态硬边界 PINN 训练器（V6 两阶段修订）

相对旧版的改动：
  1) 两阶段课程：σ 表示课程(Phase0，model.set_sigma 余弦退火) + β 加权课程(可阶跃/余弦，可绑定 Gate)。
  2) Gate 门控：用【不依赖真解】的信号判定进入 Phase1 —— 训练损失平台 + 残差峰值位置稳定
     + 参数漂移增速回落；通过后自动冻结残差对比度场(冻结 W)，替代旧版固定 warmstart_epochs。
  3) 空间缓变权重场：按 update_field_freq 步更新 criterion 的 EMA 残差场（抑制自适应权重锯齿）。
  4) LBFGS 安全化：仅允许在权重冻结态运行（state-dependent 权重 + LBFGS 线搜索是旧版 nan 的根因）。
  5) 梯度范数改为 backward 之后记录“当前步”梯度（旧版记录的是上一步）。
  6) 动力学新增 focus_ratio=带内/带外权重均值比；best checkpoint 判据可选 max(全局L2,带内L2)。
  7) 日志文件按 run_tag 隔离，避免多跑混写。
向后兼容：不配置新课程/Gate 时，行为与旧版一致（仍可用 warmstart_epochs 固定步冻结）。
"""

import json, os, copy, numpy as np, torch
from datetime import datetime
from utils.metrics import PINNMetrics


class GateDetector:
    """Phase0->Phase1 门控（全部信号不依赖真解，可实现）。"""
    def __init__(self, nu, patience=3, peak_tol=5.0, plateau_rel=0.15, drift_decay=0.95):
        self.nu = nu
        self.patience = int(patience)
        self.peak_tol = float(peak_tol) * nu     # 残差峰值位置稳定阈值（按 ν 缩放）
        self.plateau_rel = float(plateau_rel)
        self.drift_decay = float(drift_decay)
        self.hist_peak, self.hist_loss, self.hist_dspeed = [], [], []
        self._prev_drift, self._prev_step = 0.0, 0
        self.ok_streak = 0
        self.passed = False

    def update(self, epoch, loss_val, x_peak, param_drift):
        dspeed = abs(param_drift - self._prev_drift) / max(epoch - self._prev_step, 1)
        self._prev_drift, self._prev_step = param_drift, epoch
        self.hist_peak.append(x_peak); self.hist_loss.append(loss_val); self.hist_dspeed.append(dspeed)
        ok = True
        if len(self.hist_peak) >= 2:
            ok &= abs(self.hist_peak[-1] - self.hist_peak[-2]) <= self.peak_tol
        else:
            ok = False
        if len(self.hist_loss) >= 2 and self.hist_loss[-2] > 1e-12:
            ok &= abs(self.hist_loss[-1] - self.hist_loss[-2]) / self.hist_loss[-2] <= self.plateau_rel
        # 参数漂移增速需回落（相对最近峰值）
        if len(self.hist_dspeed) >= 4:
            recent = self.hist_dspeed[-1]; peak = max(self.hist_dspeed[:-1] + [1e-12])
            ok &= recent <= self.drift_decay * peak
        else:
            ok = False
        self.ok_streak = self.ok_streak + 1 if ok else 0
        if self.ok_streak >= self.patience:
            self.passed = True
        return self.passed


class PINNTrainer:
    def __init__(self, model, pde_engine, criterion, optimizer, sampler, visualizer, device, config, test_data=None):
        self.model = model
        self.pde_engine = pde_engine
        self.criterion = criterion
        self.optimizer = optimizer
        self.sampler = sampler
        self.visualizer = visualizer
        self.device = device
        self.config = config
        self.loss_history, self.l2_history = [], []
        self.nu = float(config.get("nu", 0.01))

        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.eval_freq = int(config.get("eval_freq", 500))
        self.warmstart_epochs = int(config.get("warmstart_epochs", 0))
        self.use_best_ckpt = bool(config.get("use_best_ckpt", True))
        self.best_metric = config.get("best_metric", "L2")            # 'L2' 或 'max_L2_band'
        self.shock_band_c = float(config.get("shock_band_c", 3.0))

        # σ 表示课程
        self.sigma_lo = config.get("sigma_lo", None)
        self.sigma_hi = config.get("sigma_hi", None)
        self.sigma_T = int(config.get("sigma_anneal_T", 0))
        self.sigma_fix = float(config.get("fourier_scale", 1.0))

        # β 加权课程
        self.beta_schedule = config.get("beta_schedule", "const")    # const/step/cosine
        self.beta_target = float(config.get("loss_beta", 1.0))
        self.beta_init = float(config.get("beta_init", 1.0))
        self.beta_start = int(config.get("beta_start_step", 0))
        self.beta_end = int(config.get("beta_end_step", 0))
        self.gate_epoch = -1

        # 空间缓变权重场
        self.field_update_freq = int(config.get("update_field_freq", 100))
        self.spatial_ema = float(config.get("spatial_ema_decay", 0.0)) > 0

        # Gate
        self.use_gate = bool(config.get("use_gate", False))
        self.gate = GateDetector(self.nu,
                                 patience=config.get("gate_patience", 3),
                                 peak_tol=config.get("gate_peak_tol", 5.0),
                                 plateau_rel=config.get("gate_plateau_rel", 0.15)) if self.use_gate else None
        self.gate_lr_gamma = float(config.get("gate_lr_gamma", 1.0))

        tag = config.get("run_tag", "run")
        # 日志/产物目录可由实验脚本通过 config['out_dir'] 指定（每实验独立文件夹）；
        # 未指定时退回 ./results，保持向后兼容。
        self.out_dir = config.get("out_dir", "./results")
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.out_dir, f'training_log_{tag}.txt'), 'a', encoding='utf-8')
        self.log_file.write("Epoch | Total_Loss | L2_Error | L_inf_Error\n" + "-" * 50 + "\n"); self.log_file.flush()

        self.dynamics_history = {
            'step': [], 'total_loss': [], 'loss_shock': [], 'loss_smooth': [],
            'grad_norm': [], 'weight_mean_shock': [], 'weight_mean_smooth': [],
            'weight_band_max': [], 'focus_ratio': [], 'grad_energy_ratio': [],
            'ntk_trace_ratio': [], 'param_drift': [], 'sigma': [], 'beta': [],
            'L2_traj': [], 'Linf_traj': [], 'l2_band_traj': [], 'eval_step': [],
        }

        self.theta0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.theta0_norm = float(torch.sqrt(sum((v.detach() ** 2).sum() for v in model.parameters())))

        self.best_state = copy.deepcopy(model.state_dict())
        self.best_score, self.best_L2, self.best_epoch = float('inf'), float('inf'), -1

        spectral_n = config.get('spectral_n_points', 800)
        spectral_data = sampler.sample(spectral_n, mode='random')
        self._spectral_batch = spectral_data['x_pde'].detach().to(device)
        print(f"[Spectral] Fixed subset size: {spectral_n}")

    # ---------------- 课程调度 ----------------
    def _sigma_at(self, ep):
        if self.sigma_lo is None or self.sigma_T <= 0:
            return self.sigma_fix
        if ep >= self.sigma_T:
            return float(self.sigma_hi)
        q = ep / self.sigma_T
        return float(self.sigma_lo + (self.sigma_hi - self.sigma_lo) * 0.5 * (1 - np.cos(np.pi * q)))

    def _beta_at(self, ep):
        if self.beta_schedule == "const":
            return self.beta_target
        start = self.gate_epoch if (self.beta_schedule == "gate" and self.gate_epoch >= 0) else self.beta_start
        if self.beta_schedule == "step":
            return self.beta_init if ep < start else self.beta_target
        if self.beta_schedule in ("cosine", "gate"):
            if ep < start: return self.beta_init
            end = start + (self.beta_end - self.beta_start) if self.beta_schedule == "gate" else self.beta_end
            if end <= start or ep >= end: return self.beta_target
            q = (ep - start) / max(end - start, 1)
            return self.beta_init + (self.beta_target - self.beta_init) * 0.5 * (1 - np.cos(np.pi * q))
        return self.beta_target

    # ---------------- 入口 ----------------
    def train(self, config, test_data, adam_epochs, lbfgs_epochs, adaptive_freq):
        print(f"\n--- [Phase Adam] ({adam_epochs})  schedule: σ={self.sigma_lo}->{self.sigma_hi} "
              f"β({self.beta_schedule})->{self.beta_target} gate={self.use_gate} ---")
        milestones = config.get("lr_milestones", None)
        if milestones:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=float(config.get("lr_gamma", 0.3)))
        elif config.get("lr_cosine", False):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=adam_epochs)
        else:
            self.scheduler = None
        self._train_adam(config, test_data, adam_epochs, adaptive_freq)

        if lbfgs_epochs and lbfgs_epochs > 0:
            frozen = getattr(self.criterion, "field_frozen", False) or self.criterion.weight_mode != "adaptive"
            if not frozen:
                print("[LBFGS] 自适应权重未冻结，先冻结残差场以防线搜索非光滑导致 nan")
                self.criterion.freeze_residual_field(self.model, self.pde_engine)
            print(f"\n--- [Phase L-BFGS] ({lbfgs_epochs}) 权重已冻结 ---")
            if self.sampler.use_adaptive:
                final_data = self.sampler.sample(config['n_pde'], mode='random')
                final_res = self.pde_engine.compute_residual(self.model, final_data['x_pde'])
                self.sampler.update_buffer(final_data['x_pde'].detach(), final_res.abs().detach())
                self.sampler.fixed_points = None
            lbfgs_optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=float(config.get("lbfgs_lr", 1.0)), max_iter=20,
                tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)
            self._train_lbfgs(config, test_data, lbfgs_optimizer, lbfgs_epochs)

        if self.use_best_ckpt and self.best_epoch >= 0:
            self.model.load_state_dict(self.best_state)
            print(f"[BestCkpt] restored best @epoch {self.best_epoch}, L2={self.best_L2:.4e}")

    def _param_drift(self):
        if self.theta0_norm < 1e-12:
            return 0.0
        s = torch.zeros((), device=self.device)
        sd = self.model.state_dict()
        for k, v0 in self.theta0.items():
            if k in sd and sd[k].dtype.is_floating_point:
                s = s + ((sd[k].to(v0) - v0) ** 2).sum()
        return float(torch.sqrt(s) / self.theta0_norm)

    def _eval_test(self, test_data):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(test_data['x']).detach().cpu().numpy().flatten()
        m = PINNMetrics.compute_fast(u_pred=u_pred, u_true=test_data['u_true'])
        self.model.train()
        return m

    def _residual_peak(self):
        """残差峰值位置（Gate 用，不依赖真解）：在固定 ref grid 上取 |r| argmax。
        残差需二阶 autograd，不能用 no_grad 包裹，结果取出后立即 detach。"""
        g = self.criterion._ref_grid.view(-1, 1)
        with torch.enable_grad():
            r = self.pde_engine.compute_residual(self.model, g).reshape(-1).abs().detach()
        return float(self.criterion._ref_grid[int(torch.argmax(r))])

    def _train_adam(self, config, test_data, epochs, adaptive_freq):
        self.model.train()
        band = self.shock_band_c * self.nu
        log_freq = config.get('dynamics_log_freq', 200)
        dh = self.dynamics_history

        for epoch in range(epochs):
            # 课程调度
            sig = self._sigma_at(epoch); self.model.set_sigma(sig)
            beta = self._beta_at(epoch); self.criterion.set_beta(beta)

            data = self.sampler.sample(config['n_pde'], mode='random')
            x_pde = data['x_pde']
            pde_res = self.pde_engine.compute_residual(self.model, x_pde)

            if self.sampler.use_adaptive and (epoch % adaptive_freq == 0):
                self.sampler.update_buffer(x_pde.detach(), pde_res.abs().detach())

            # 空间缓变权重场更新（Phase0 未冻结时）
            if self.spatial_ema and not getattr(self.criterion, "field_frozen", False) \
                    and self.criterion.weight_mode == "adaptive" and epoch % self.field_update_freq == 0:
                self.criterion.update_residual_field(self.model, self.pde_engine)

            # 旧版固定步 warmstart（向后兼容）
            if (self.criterion.weight_mode == "warmstart_spatial"
                    and not self.criterion.warmstart_frozen
                    and epoch == self.warmstart_epochs):
                lo, hi = self.criterion.freeze_warmstart(self.model)
                print(f"[WarmStart] frozen @{epoch} w=[{lo:.3f},{hi:.3f}]")

            total_loss, _ = self.criterion(pde_res, x_pde=x_pde)

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # 动力学采集（grad_norm 现在是“当前步”梯度，在 backward 之后）
            if epoch % log_freq == 0:
                grad_sq = [p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None]
                grad_norm = float(torch.sqrt(torch.stack(grad_sq).sum())) if grad_sq else 0.0
                loss_shock = loss_smooth = w_in = w_out = w_bmax = focus = 0.0
                ge_ratio = 0.0
                res_abs = pde_res.detach().reshape(-1)
                with torch.no_grad():
                    w = self.criterion.get_current_weights(res_abs, x_pde=x_pde.detach())
                    bm = (x_pde.detach().reshape(-1).abs() < band)
                    if bm.any():
                        rb = res_abs[bm]
                        loss_shock = float((w[bm] * rb ** 2).mean()); w_in = float(w[bm].mean()); w_bmax = float(w[bm].max())
                    sm = ~bm
                    if sm.any():
                        loss_smooth = float((w[sm] * res_abs[sm] ** 2).mean()); w_out = float(w[sm].mean())
                    focus = w_in / max(w_out, 1e-12)
                try:
                    def grad_energy(xs):
                        if xs.shape[0] == 0: return 0.0
                        with torch.enable_grad():
                            xi = xs.detach().requires_grad_(True)
                            out = self.model(xi)
                            gg = torch.autograd.grad(out, xi, torch.ones_like(out), create_graph=False)[0]
                            return float((gg.detach() ** 2).sum())
                    ge_ratio = grad_energy(x_pde[bm]) / (grad_energy(x_pde[~bm]) + 1e-12)
                except Exception as e:
                    print(f"  [GradEnergy] fail: {e}")

                drift = self._param_drift()
                dh['step'].append(epoch); dh['total_loss'].append(float(total_loss.detach()))
                dh['loss_shock'].append(loss_shock); dh['loss_smooth'].append(loss_smooth)
                dh['grad_norm'].append(grad_norm)
                dh['weight_mean_shock'].append(w_in); dh['weight_mean_smooth'].append(w_out)
                dh['weight_band_max'].append(w_bmax); dh['focus_ratio'].append(focus)
                dh['grad_energy_ratio'].append(ge_ratio); dh['ntk_trace_ratio'].append(ge_ratio)
                dh['param_drift'].append(drift); dh['sigma'].append(sig); dh['beta'].append(beta)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.loss_history.append(float(total_loss.detach()))

            # 评估 + Gate
            if epoch % self.eval_freq == 0 or epoch == epochs - 1:
                fm = self._eval_test(test_data)
                l2b = self._band_l2(test_data)
                dh['eval_step'].append(epoch); dh['L2_traj'].append(fm['L2_Error'])
                dh['Linf_traj'].append(fm['L_inf_Error']); dh['l2_band_traj'].append(l2b)
                score = max(fm['L2_Error'], l2b) if self.best_metric == "max_L2_band" else fm['L2_Error']
                if np.isfinite(score) and score < self.best_score:
                    self.best_score, self.best_L2, self.best_epoch = score, fm['L2_Error'], epoch
                    self.best_state = copy.deepcopy(self.model.state_dict())

                if self.use_gate and not self.gate.passed and self.criterion.weight_mode == "adaptive":
                    x_peak = self._residual_peak()
                    if self.gate.update(epoch, float(total_loss.detach()), x_peak, self._param_drift()):
                        self.gate_epoch = epoch
                        lo, hi = self.criterion.freeze_residual_field(self.model, self.pde_engine)
                        if self.gate_lr_gamma < 1.0:
                            for g in self.optimizer.param_groups: g['lr'] *= self.gate_lr_gamma
                        print(f"[Gate] PASSED @{epoch}: 冻结W w=[{lo:.3f},{hi:.3f}], 进入 Phase1, gate_lr×{self.gate_lr_gamma}")

                if epoch % 1000 == 0:
                    self._log_metrics(epoch, total_loss, test_data)

    def _band_l2(self, test_data):
        self.model.eval()
        with torch.no_grad():
            up = self.model(test_data['x']).detach().cpu().numpy().flatten()
        xn = test_data['x'].detach().cpu().numpy().flatten()
        ut = np.asarray(test_data['u_true']).flatten()
        m = np.abs(xn) < self.shock_band_c * self.nu
        self.model.train()
        return float(np.linalg.norm((up - ut)[m]) / max(np.linalg.norm(ut[m]), 1e-8)) if m.any() else 0.0

    def _train_lbfgs(self, config, test_data, lbfgs_optimizer, epochs):
        self.model.train()
        data = self.sampler.sample(config['n_pde'], mode='fixed')
        for epoch in range(epochs):
            def closure():
                lbfgs_optimizer.zero_grad()
                pde_res = self.pde_engine.compute_residual(self.model, data['x_pde'])
                loss, _ = self.criterion(pde_res, x_pde=data['x_pde'])
                loss.backward()
                return loss
            total_loss = lbfgs_optimizer.step(closure)
            self.loss_history.append(float(total_loss))
            fm = self._eval_test(test_data)
            score = fm['L2_Error']
            if np.isfinite(score) and score < self.best_score:
                self.best_score, self.best_L2, self.best_epoch = score, fm['L2_Error'], epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
            if epoch % 100 == 0:
                self._log_metrics(epoch, total_loss, test_data)

    def _log_metrics(self, epoch, total_loss, test_data, pde_res=None):
        fm = self._eval_test(test_data)
        self.l2_history.append(fm["L2_Error"])
        tl = float(total_loss.detach()) if torch.is_tensor(total_loss) else float(total_loss)
        log_str = f"Epoch {epoch:04d} | Loss {tl:.4e} | L2 {fm['L2_Error']:.4e} | Linf {fm['L_inf_Error']:.4e}"
        print(log_str)
        self.log_file.write(log_str + "\n"); self.log_file.flush()

    def evaluate(self, test_data):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(test_data['x']).detach().cpu().numpy().flatten()
        pde_res = self.pde_engine.compute_residual(self.model, test_data['x'])
        weights = np.ones_like(u_pred)
        if hasattr(self.criterion, 'get_current_weights'):
            weights = self.criterion.get_current_weights(
                pde_res.detach(), x_pde=test_data['x'].detach()
            ).detach().cpu().numpy().flatten()
        pde_res_np = pde_res.detach().cpu().numpy().flatten()
        x_np = test_data['x'].detach().cpu().numpy()
        metrics = PINNMetrics.compute_full(
            u_pred=u_pred, u_true=test_data['u_true'], x=x_np,
            pde_res=pde_res_np, weights=weights, nu=self.nu,
            shock_band_c=self.shock_band_c)
        metrics['best_epoch'] = int(self.best_epoch)
        metrics['gate_epoch'] = int(self.gate_epoch)
        metrics['param_drift'] = float(self._param_drift())
        return metrics

    def save_solution(self, x_grid, u_true, save_path):
        self.model.eval()
        with torch.no_grad():
            if isinstance(x_grid, np.ndarray):
                x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=self.device)
            else:
                x_tensor = x_grid.to(self.device)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(1)
            u_pred = self.model(x_tensor).cpu().numpy().flatten()
        u_true_np = u_true.flatten() if isinstance(u_true, np.ndarray) else u_true.detach().cpu().numpy().flatten()
        x_np = x_grid.flatten() if isinstance(x_grid, np.ndarray) else x_grid.detach().cpu().numpy().flatten()
        solution_data = {
            "x": x_np.tolist(), "u_true": u_true_np.tolist(), "u_pred": u_pred.tolist(),
            "metadata": {
                "nu": self.config['nu'], "beta": self.config['loss_beta'],
                "weight_mode": self.criterion.weight_mode, "best_epoch": self.best_epoch,
                "gate_epoch": self.gate_epoch,
                "num_points": len(x_np), "timestamp": datetime.now().isoformat(),
            },
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(solution_data, f)
        print(f"[Solution Saved] {save_path} ({len(x_np)} points)")
