# -*- coding: utf-8 -*-
"""
PINN 评估指标（修订版）
口径说明（务必在论文中注明）：
  L2_Error   : 相对误差 ||u_pred-u_true|| / ||u_true||
  L_inf_Error: 绝对误差 max|u_pred-u_true|（未归一化，激波跳变≈2，故其值可接近 2）
  分区统一用“固定激波带 |x| < shock_band_c*ν”（按坐标，可复现、不随残差变化）；
  Oscillation_Count 先做 3 点平滑再数梯度变号，抑制高频噪声过敏。
"""

import numpy as np


class PINNMetrics:
    # ---------------- 轻量（训练中） ----------------
    @staticmethod
    def compute_fast(pde_res=None, weights=None, u_pred=None, u_true=None):
        m = {'L2_Error': 0.0, 'L_inf_Error': 0.0, 'loss_shock': 0.0, 'loss_smooth': 0.0}
        if u_pred is not None and u_true is not None:
            diff = np.asarray(u_pred) - np.asarray(u_true)
            m['L2_Error'] = float(np.linalg.norm(diff) / max(np.linalg.norm(u_true), 1e-12))
            m['L_inf_Error'] = float(np.max(np.abs(diff)))
        return m

    # ---------------- 完整（终评估） ----------------
    @staticmethod
    def compute_full(u_pred, u_true, x=None, nu=0.01, pde_res=None, weights=None,
                     shock_band_c=3.0, smooth_win=3):
        m = {}
        u_pred = np.asarray(u_pred).ravel(); u_true = np.asarray(u_true).ravel()
        diff = u_pred - u_true
        m["L2_Error"] = float(np.linalg.norm(diff) / max(np.linalg.norm(u_true), 1e-12))
        m["L_inf_Error"] = float(np.max(np.abs(diff)))

        if x is None:
            return m
        x = np.asarray(x).ravel()
        order = np.argsort(x); x = x[order]; u_pred = u_pred[order]; u_true = u_true[order]
        diff = u_pred - u_true
        if len(x) <= 2:
            return m
        dx = np.diff(x)

        # TV 相对误差
        tv_pred = np.sum(np.abs(np.diff(u_pred))); tv_true = np.sum(np.abs(np.diff(u_true)))
        m["TV_Error"] = float(abs(tv_pred - tv_true) / (tv_true + 1e-12))

        # 振荡计数：先轻平滑再数梯度变号
        def _smooth(a, w=3):
            if w <= 1 or len(a) < w: return a
            k = np.ones(w) / w
            return np.convolve(a, k, mode='same')
        g_sm = np.diff(_smooth(u_pred, smooth_win)) / (dx + 1e-12)
        m["Oscillation_Count"] = int(np.sum(np.diff(np.sign(g_sm)) != 0) // 2)

        # 激波宽度比 / 有效粘性比（稳健穿越点口径，不要求预测剖面全局单调）
        # 旧版用 np.interp(th, -u_pred, x)，但坏盆地剖面非单调时 -u_pred 不递增、结果错误；
        # 改为“高位簇右沿 / 低位簇左沿”的穿越点估计，对弱非单调稳健。
        umax, umin = u_true.max(), u_true.min()
        th_hi = umin + 0.9 * (umax - umin); th_lo = umin + 0.1 * (umax - umin)
        ih = np.where(u_true >= th_hi)[0]; il = np.where(u_true <= th_lo)[0]
        iph = np.where(u_pred >= th_hi)[0]; ipl = np.where(u_pred <= th_lo)[0]
        if len(ih) and len(il):
            w_true = abs(x[il[0]] - x[ih[-1]])
            if len(iph) and len(ipl):
                xh = x[iph[-1]]   # 高位区(u>=th_hi)右沿
                xl = x[ipl[0]]    # 低位区(u<=th_lo)左沿
                w_pred = abs(xl - xh)
                m["Shock_Width_Ratio"] = float(w_pred / (w_true + 1e-12))
                coeff = 4.0 * np.arctanh(0.8)
                m["Nu_Eff_Ratio"] = float((w_pred / coeff) / (nu + 1e-12))
                # 剖面非单调标志：排序后预测仍出现显著反向变化 => 坏盆地/振荡
                du = np.diff(u_pred)
                m["nonmonotone_frac"] = float(np.mean(du > 1e-3))
            else:
                m["Shock_Width_Ratio"] = 1.0; m["Nu_Eff_Ratio"] = 1.0; m["nonmonotone_frac"] = 1.0

        # 固定激波带（按坐标）
        band = shock_band_c * nu
        shock = np.abs(x) < band
        smooth = ~shock

        if pde_res is not None:
            pde_res = np.asarray(pde_res).ravel()[order]
            res_sq = pde_res ** 2
            m['loss_shock'] = float(np.mean(res_sq[shock])) if shock.any() else 0.0
            m['loss_smooth'] = float(np.mean(res_sq[smooth])) if smooth.any() else 0.0
            m['max_residual'] = float(np.max(np.abs(pde_res)))
            tot = np.sum(np.abs(pde_res)) + 1e-12
            m['shock_residual_ratio'] = float(np.sum(np.abs(pde_res[shock])) / tot) if shock.any() else 0.0

        # 分区相对 L2 + 固定带内 L2
        m['l2_shock'] = float(np.linalg.norm(diff[shock]) / max(np.linalg.norm(u_true[shock]), 1e-8)) if shock.any() else 0.0
        m['l2_smooth'] = float(np.linalg.norm(diff[smooth]) / max(np.linalg.norm(u_true[smooth]), 1e-8)) if smooth.any() else 0.0
        m['l2_band'] = m['l2_shock']

        if weights is not None:
            weights = np.asarray(weights).ravel()[order]
            m['weight_mean_shock'] = float(np.mean(weights[shock])) if shock.any() else 0.0
            m['weight_max_shock'] = float(np.max(weights[shock])) if shock.any() else 0.0

        # 原生类型化，防 JSON 序列化问题
        m = {k: (float(v) if isinstance(v, (float, np.floating)) else
                 int(v) if isinstance(v, (int, np.integer)) else v) for k, v in m.items()}
        return m
