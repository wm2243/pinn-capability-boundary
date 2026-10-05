# -*- coding: utf-8 -*-
"""
实验 G2（V9.3 升级）：无真解盆地可判别性 Gate-B —— 单标量必败 vs 多变量结构量可分
================================================================================
对应 framework_v9_3 命题 8.8（Bayes 必要条件）/ 注记 8.10（Cramér-Wold 划界）/ 定义 5.7（Gate-B）。
要回答三件事：
  (1) 命题 8.8(ii)：每个【预先指定的单标量】g1/g2/g3（损失 RMS、残差水平、慢模态能量等幅值坐标）
      在好/坏盆地的边缘是否近乎重合？定量给 TV / KS（即命题里的 τ=max_k TV_k），单标量 gate 是否退化为掷硬币。
  (2) 命题 8.8(iii)：固定单坐标全重叠不蕴含联合不可分——多变量【结构/方向量】（残差峰位置、带内质量、
      冻结 Hessian 负曲率维数、多起点离散度等）经留一交叉验证能否显著区分？给 sens/spec/AUC。
  (3) 数据泄漏修正（V9.3 严谨性）：无真解部署时只能按 proxy（末段损失）选起点，故【主口径】对 proxy 选出的
      起点 pbest 提特征、并用其真解 L2 事后贴标签；对 truth 选起点 best 的判别仅作【oracle 上界】单列，
      不进主判别器。agree（proxy/truth 选点是否一致）依赖 truth，移出特征、仅作事后列。

纯 numpy（不依赖 sklearn）：Mann-Whitney AUC、Youden、留一；TV/KS；L2 逻辑回归（手写梯度）多变量留一。
四点坐标例自检（注记 8.10）：μG=同号、μB=异号，单坐标 AUC=0.5、纯线性 (Z1,Z2) 学 XOR≈0.5、
      加结构量 Z1Z2 后 AUC=1，精确印证“单标量必败、需多变量结构量、且难点在不知看哪个结构方向”。

产出 results/v6/exp_G2_diagnose/：
  g2_samples.csv      主口径（proxy 选起点）每 (nu,seed)：无真解特征 + 事后 truth 标签 + oracle 列
  g2_feature_auc.csv  每单标量 AUC/方向/Youden/留一 + TV + KS（命题(ii) 的 τ）
  g2_group_loo.csv    MAG(幅值组)/STRUCT(结构组)/ALL 多变量逻辑回归留一 sens/spec/acc/auc
  g2_oracle_upper.csv truth 选起点的同流程（上界参考，量化“选起点这步的损失”）
  g2_toy_check.csv    四点坐标例自检
  g2_pseudo.csv       主口径伪好/伪坏及逐特征 z 分
  fig_g2_box / roc / scatter / tvks / groups .png
环境：V8_SMOKE=1 自检；G2_K 多起点（默认4）；G2_HESS=0 关冻结 Hessian（默认开）。
运行：python exp_G2_basin_diagnose.py
"""
import os, sys, csv, copy, math
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from trainers.pinn_trainer import PINNTrainer

OUT = str(mc.RES_ROOT / "exp_G2_diagnose"); os.makedirs(OUT, exist_ok=True); r2s.OUT = OUT
SAMPLES_CSV = os.path.join(OUT, "g2_samples.csv")
NUS = [0.01, 0.005, 0.003, 0.001]
SEEDS = list(range(0, 100, 5))          # 与 G1 扩样一致：规则等距、实验前固定，K=4 内部 s..s+3 不重叠
EP0 = 2000
K_START = int(os.environ.get("G2_K", "4"))
USE_HESS = os.environ.get("G2_HESS", "1") == "1"
if mc.SMOKE: NUS, SEEDS, EP0, K_START = [0.005], [0], 240, 1

# 幅值/单标量组（命题 8.8(ii) 的“固定标量字典”，g1/g2/g3 类：损失/残差/能量的水平值）
MAG = ["r_resid", "focus_t", "focus_cv", "ge_t", "wmean_t", "gradnorm_t",
       "loss_slope", "loss_cv", "proxy"]
# 结构/方向量组（命题 8.8(iii)：位置、形状、曲率、离散度等结构量）
STRUCT = ["abs_peak", "peak_in_band", "peak_ratio", "lam_min", "n_neg",
          "logkappa", "gate_frac", "starts_cv"]
FEATURES = MAG + STRUCT                # 全部【无真解】特征；agree_select 已移出（泄漏）
GROUPS = {"MAG_幅值单标量": MAG, "STRUCT_结构量": STRUCT, "ALL_全多变量": FEATURES}
FEATURE_NAME = dict(abs_peak="残差峰|x|", peak_in_band="带内残差质量比", peak_ratio="残差峰/均值",
                    r_resid="残差 RMS", focus_t="末段聚焦度", focus_cv="聚焦度变异",
                    ge_t="梯度能量比", wmean_t="激波均权", gradnorm_t="梯度范数",
                    loss_slope="损失末段斜率", loss_cv="损失末段变异", lam_min="Hessian λmin",
                    n_neg="负曲率维数", logkappa="log κ", proxy="末段损失", gate_frac="Gate 相对位置",
                    starts_cv="多起点离散")


# ---------- 带轨迹的 Phase0（复刻 r2s.run_phase0，额外回收 dyn/loss_hist，不改公共代码）----------
def phase0_with_traj(nu, base_seed, xt, ut):
    r2s.MULTISTART_K = K_START; r2s.SELECT_BY = "truth"
    r2s.SIG_HI = r2s.SIGMA_HI_CFG; r2s.PHASE0_EPOCHS = EP0
    cands = []
    for k in range(K_START):
        seed = base_seed + k; r2s.set_seed(seed)
        cfg = r2s.base_cfg(nu, f"g2_{base_seed}_{k}")
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG, sigma_anneal_T=r2s.SIGMA_T,
                        beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=EP0, lbfgs_epochs=0, use_best_ckpt=True, best_metric="max_L2_band"))
        model, pde, crit0, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE0, 1.0)
        tr = PINNTrainer(model, pde, crit0, opt, samp, vis, mc.DEVICE, cfg, {"x": xt, "u_true": ut})
        tr.train(cfg, {"x": xt, "u_true": ut}, EP0, 0, adaptive_freq=500)
        if not crit0.field_frozen: crit0.freeze_residual_field(model, pde)
        l2 = r2s.quick_l2(model, xt, ut)
        cands.append(dict(seed=seed, l2=l2, proxy=float(np.mean(tr.loss_history[-200:])),
                          gate=tr.gate_epoch, state=copy.deepcopy(model.state_dict()),
                          t_field=crit0._frozen_t_field.detach().clone(), sigma_end=model.get_sigma(),
                          dyn=copy.deepcopy(tr.dynamics_history), loss_hist=list(tr.loss_history),
                          model=model, pde=pde, crit=crit0))
        tr.log_file.close()
    best = min(cands, key=lambda z: z["l2"])            # oracle：真解选盆地（仅作上界）
    pbest = min(cands, key=lambda z: z["proxy"])        # 部署口径：无真解只能按 proxy 选
    return best, pbest, cands


def _terminal(seq, k=5):
    a = np.asarray(seq, float); a = a[np.isfinite(a)]
    return a[-k:] if a.size >= 1 else np.asarray([np.nan])


def _cv(a):
    a = np.asarray(a, float); m = np.nanmean(a)
    return float(np.nanstd(a) / abs(m)) if np.isfinite(m) and abs(m) > 1e-30 else float("nan")


def extract_features(cand, cands):
    """对指定起点 cand 提【无真解】特征；真解只用于事后标签。"""
    model, pde, crit = cand["model"], cand["pde"], cand["crit"]
    xt = torch.linspace(-1, 1, 2048, device=mc.DEVICE).view(-1, 1).requires_grad_(True)
    r = pde.compute_residual(model, xt).detach().cpu().numpy().reshape(-1)
    x = xt.detach().cpu().numpy().reshape(-1)
    nu = float(crit.nu); band = 3.0 * nu; inb = np.abs(x) < band
    ar = np.abs(r)
    f = dict(abs_peak=float(abs(x[int(np.argmax(ar))])),
             peak_in_band=float(ar[inb].sum() / max(ar.sum(), 1e-30)) if inb.any() else np.nan,
             peak_ratio=float(ar.max() / max(ar.mean(), 1e-30)), r_resid=float(np.sqrt(np.mean(r ** 2))))
    dyn = cand["dyn"]
    for key, col in [("focus_t", "focus_ratio"), ("ge_t", "grad_energy_ratio"),
                     ("wmean_t", "weight_mean_shock"), ("gradnorm_t", "grad_norm")]:
        f[key] = float(np.nanmean(_terminal(dyn.get(col, []))))
    f["focus_cv"] = _cv(_terminal(dyn.get("focus_ratio", [])))
    lh = np.asarray(cand["loss_hist"], float); tail = lh[-200:] if lh.size >= 200 else lh
    f["loss_slope"] = float(np.polyfit(np.arange(tail.size), tail / max(abs(tail[0]), 1e-30), 1)[0]) if tail.size > 5 else np.nan
    f["loss_cv"] = _cv(tail); f["proxy"] = cand["proxy"]
    f["gate_frac"] = float(cand["gate"] / EP0) if cand["gate"] and cand["gate"] >= 0 else 0.0
    if USE_HESS:
        h = mc.hessian_min_curvature(model, pde, crit, xt, num_lanczos=40)
        f["lam_min"] = h["lam_min"]; f["n_neg"] = float(h["n_neg"])
        f["logkappa"] = float(np.log10(max(h["kappa"], 1e-12))) if np.isfinite(h["kappa"]) else np.nan
    else:
        f["lam_min"] = f["n_neg"] = f["logkappa"] = np.nan
    proxies = np.asarray([c["proxy"] for c in cands], float)
    f["starts_cv"] = float(proxies.std() / abs(proxies.mean())) if proxies.size > 1 and proxies.mean() != 0 else 0.0
    # 真解仅用于事后标签 / 事后列（绝不进 FEATURES）
    f["truth_L2"] = cand["l2"]; f["truth_good"] = int(cand["l2"] < mc.GATE_L2_MAX)
    return f


def collect():
    done, rows, rows_oracle = {}, [], []
    if os.path.exists(SAMPLES_CSV):
        with open(SAMPLES_CSV, "r", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh): rows.append(r); done[(r["nu"], r["seed"])] = r
    for nu in NUS:
        xt, ut = r2s.build_test(nu)
        for seed in SEEDS:
            tag = (f"{nu:g}", str(seed))
            if tag in done: continue
            best, pbest, cands = phase0_with_traj(nu, seed, xt, ut)
            f = extract_features(pbest, cands)          # 主口径：proxy 选起点（无真解部署）
            fo = extract_features(best, cands)          # oracle 上界：truth 选起点
            f.update(nu=nu, seed=seed, select_agree=int(best["seed"] == pbest["seed"]),
                     oracle_L2=best["l2"], oracle_good=int(best["l2"] < mc.GATE_L2_MAX))
            fo.update(nu=nu, seed=seed)
            rows.append(f); rows_oracle.append(fo); _dump(rows)
            print(f"[G2] nu={nu:g} s{seed}: 部署(proxy)good={f['truth_good']} "
                  f"oracleGood={fo['truth_good']} agree={f['select_agree']} |peak|={f['abs_peak']:.3f}")
    for r in rows:
        for c in FEATURES + ["truth_L2", "oracle_L2"]: r[c] = float(r[c])
        r["truth_good"] = int(float(r["truth_good"])); r["oracle_good"] = int(float(r["oracle_good"]))
        r["select_agree"] = int(float(r["select_agree"])); r["seed"] = int(float(r["seed"])); r["nu"] = float(r["nu"])
    for r in rows_oracle:
        for c in FEATURES + ["truth_L2"]: r[c] = float(r[c])
        r["truth_good"] = int(float(r["truth_good"])); r["seed"] = int(float(r["seed"])); r["nu"] = float(r["nu"])
    return rows, rows_oracle


COLS = ["nu", "seed"] + FEATURES + ["truth_L2", "truth_good", "select_agree", "oracle_L2", "oracle_good"]


def _dump(rows):
    mc.write_csv(SAMPLES_CSV, [{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows], COLS)


# ---------- 分布距离 TV / KS（命题 8.8(ii) 的 τ）----------
def tv_ks(score, y, n_bin=10):
    s = np.asarray(score, float); ok = np.isfinite(s)
    s = np.where(ok, s, np.nanmedian(s[ok]) if ok.any() else 0.0); y = np.asarray(y)
    g, b = s[y == 1], s[y == 0]
    if g.size == 0 or b.size == 0: return float("nan"), float("nan")
    # KS：经验 CDF 最大差
    grid = np.sort(np.unique(s)); Fg = np.array([(g <= t).mean() for t in grid]); Fb = np.array([(b <= t).mean() for t in grid])
    ks = float(np.max(np.abs(Fg - Fb)))
    # TV：按全体分位边界分箱（对偏态稳健），TV=1/2 Σ|p_g−p_b|
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bin + 1))); edges[0] -= 1e-9; edges[-1] += 1e-9
    pg = np.histogram(g, bins=edges)[0] / g.size; pb = np.histogram(b, bins=edges)[0] / b.size
    tv = float(0.5 * np.abs(pg - pb).sum())
    return tv, ks


# ---------- 单特征 AUC / Youden / 留一 ----------
def clean(col, y):
    v = np.asarray(col, float); ok = np.isfinite(v)
    med = np.nanmedian(v[ok]) if ok.any() else 0.0
    return np.where(ok, v, med), np.asarray(y)


def auc_roc(score, y):
    s, y = clean(score, y); P, N = int(y.sum()), int((1 - y).sum())
    if P == 0 or N == 0: return float("nan"), None, None
    order = np.argsort(s); ss, yy = s[order], y[order]; ranks = np.empty(ss.size); i = 0
    while i < ss.size:
        j = i
        while j + 1 < ss.size and ss[j + 1] == ss[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1; i = j + 1
    a = (ranks[yy == 1].sum() - P * (P + 1) / 2) / (P * N)
    thr = np.unique(np.concatenate([[ss.min() - 1e-9], ss, [ss.max() + 1e-9]]))
    tpr = np.array([((ss >= t_) & (yy == 1)).sum() / P for t_ in thr])
    fpr = np.array([((ss >= t_) & (yy == 0)).sum() / N for t_ in thr])
    return float(a), fpr, tpr


def best_threshold(score, y, direction):
    s, y = clean(score, y); xs = s if direction >= 0 else -s
    P, N = int(y.sum()), int((1 - y).sum()); best = (-1, None)
    for t in np.unique(xs):
        pred = xs >= t; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); tn = N - fp
        j = tp / max(P, 1) + tn / max(N, 1) - 1
        if j > best[0]: best = (j, t)
    return best[1]


def analyze(rows, tag=""):
    y = np.asarray([r["truth_good"] for r in rows]); out = {}; roc = {}
    for feat in FEATURES:
        s = [r[feat] for r in rows]; a, fpr, tpr = auc_roc(s, y)
        if not np.isfinite(a): continue
        tv, ks = tv_ks(s, y); direction = 1 if a >= 0.5 else -1; a = max(a, 1 - a)
        roc[feat] = (fpr, tpr, a)
        tp = fp = tn = fn = 0; S = np.asarray(s, float)
        for i in range(len(rows)):
            tr = np.delete(np.arange(len(rows)), i)
            t = best_threshold(S[tr], y[tr], direction)
            xi = S[i] if direction >= 0 else -S[i]; pred = int(xi >= t)
            if pred and y[i] == 1: tp += 1
            elif pred and y[i] == 0: fp += 1
            elif (not pred) and y[i] == 0: tn += 1
            else: fn += 1
        out[feat] = dict(feature=feat, group=("MAG" if feat in MAG else "STRUCT"),
                         auc=float(a), TV=tv, KS=ks, direction=direction,
                         youden_thr=best_threshold(S, y, direction),
                         loo_sens=tp / max(tp + fn, 1), loo_spec=tn / max(tn + fp, 1),
                         loo_acc=(tp + tn) / max(len(rows), 1))
    table = sorted(out.values(), key=lambda z: -z["loo_acc"])
    mc.write_csv(os.path.join(OUT, f"g2_feature_auc{tag}.csv"), table)
    return out, table, roc


# ---------- 手写 L2 逻辑回归（纯 numpy），用于多变量留一 ----------
def _lr_fit(X, y, l2=1e-3, lr=0.2, it=300):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    for _ in range(it):
        z = X @ w + b; p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        e = p - y; gw = X.T @ e / n + l2 * w; gb = e.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def _standardize_fit(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1.0; return mu, sd


def group_loo(rows, tag=""):
    y = np.asarray([r["truth_good"] for r in rows]); recs = []
    for gname, feats in GROUPS.items():
        Xall = np.column_stack([clean([r[f] for r in rows], y)[0] for f in feats])
        proba = np.zeros(len(rows))
        for i in range(len(rows)):
            tr = np.delete(np.arange(len(rows)), i)
            mu, sd = _standardize_fit(Xall[tr]); Xtr = (Xall[tr] - mu) / sd; Xte = (Xall[i] - mu) / sd
            w, b = _lr_fit(Xtr, y[tr]); proba[i] = 1 / (1 + np.exp(-float(np.clip(Xte @ w + b, -30, 30))))
        a, _, _ = auc_roc(proba, y)
        thr = best_threshold(proba, y, 1); pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(y.sum()) - tp
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int((1 - y).sum()) - fp
        recs.append(dict(group=gname, n_feat=len(feats), auc=float(a),
                         loo_sens=tp / max(tp + fn, 1), loo_spec=tn / max(tn + fp, 1),
                         loo_acc=(tp + tn) / len(rows)))
    mc.write_csv(os.path.join(OUT, f"g2_group_loo{tag}.csv"), recs)
    return recs


# ---------- 四点坐标例自检（注记 8.10）----------
def toy_fourpoint(n=400, seed=0):
    rng = np.random.default_rng(seed); z = rng.choice([-1.0, 1.0], size=(n, 2))
    y = (z[:, 0] * z[:, 1] > 0).astype(int)          # 同号=好、异号=坏
    a1, _, _ = auc_roc(z[:, 0], y)                    # 单坐标 Z1
    # 纯线性 (Z1,Z2)：XOR 线性不可分
    proba_lin = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i); w, b = _lr_fit(z[tr], y[tr]); proba_lin[i] = 1 / (1 + np.exp(-(z[i] @ w + b)))
    a_lin, _, _ = auc_roc(proba_lin, y)
    # 加结构量 Z1*Z2（多变量结构/组合量）
    Zs = np.column_stack([z[:, 0], z[:, 1], z[:, 0] * z[:, 1]]); proba_s = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i); w, b = _lr_fit(Zs[tr], y[tr]); proba_s[i] = 1 / (1 + np.exp(-(Zs[i] @ w + b)))
    a_struct, _, _ = auc_roc(proba_s, y)
    tv1, ks1 = tv_ks(z[:, 0], y)
    rec = [dict(probe="单坐标 Z1（命题ii 固定标量）", auc=max(a1, 1 - a1), TV=tv1, KS=ks1),
           dict(probe="纯线性 (Z1,Z2)（线性投影/XOR）", auc=max(a_lin, 1 - a_lin), TV=float('nan'), KS=float('nan')),
           dict(probe="加结构量 Z1Z2（命题iii 多变量结构）", auc=max(a_struct, 1 - a_struct), TV=float('nan'), KS=float('nan'))]
    mc.write_csv(os.path.join(OUT, "g2_toy_check.csv"), rec)
    return rec


def combo_loo(rows, table, top=2):
    y = np.asarray([r["truth_good"] for r in rows]); picks = [t["feature"] for t in table[:top]]
    dirs = {t["feature"]: t["direction"] for t in table}; S = {f: np.asarray([r[f] for r in rows], float) for f in picks}
    res = {}
    for mode in ("AND", "OR"):
        tp = fp = tn = fn = 0
        for i in range(len(rows)):
            tr = np.delete(np.arange(len(rows)), i); votes = []
            for f in picks:
                t = best_threshold(S[f][tr], y[tr], dirs[f])
                xi = S[f][i] if dirs[f] >= 0 else -S[f][i]; votes.append(int(xi >= t))
            pred = int(all(votes)) if mode == "AND" else int(any(votes))
            if pred and y[i] == 1: tp += 1
            elif pred and y[i] == 0: fp += 1
            elif (not pred) and y[i] == 0: tn += 1
            else: fn += 1
        res[mode] = dict(sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), acc=(tp + tn) / len(rows), rule=",".join(picks))
    return res, picks, dirs, {f: best_threshold(S[f], y, dirs[f]) for f in picks}


def pseudo_report(rows, picks, dirs, thr):
    good = [r for r in rows if r["truth_good"] == 1]
    mu = {f: np.nanmean([r[f] for r in good]) for f in FEATURES}
    sd = {f: (np.nanstd([r[f] for r in good]) or np.nan) for f in FEATURES}; out = []
    for r in rows:
        pred = int(all(int((r[f] if dirs[f] >= 0 else -r[f]) >= thr[f]) for f in picks))
        if pred != r["truth_good"]:
            rec = dict(nu=r["nu"], seed=r["seed"], kind=("伪好" if pred == 1 else "伪坏"), truth_L2=r["truth_L2"])
            for f in FEATURES: rec[f"z_{f}"] = (r[f] - mu[f]) / sd[f] if np.isfinite(sd[f]) and sd[f] > 0 else np.nan
            out.append(rec)
    mc.write_csv(os.path.join(OUT, "g2_pseudo.csv"), out); return out


def plots(rows, table, roc, picks, groups):
    top = [t["feature"] for t in table[:6]]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    name_by = {t["feature"]: t for t in table}
    for k, f in enumerate(top):
        a = ax.flat[k]; g = [r[f] for r in rows if r["truth_good"] == 1]; b = [r[f] for r in rows if r["truth_good"] == 0]
        a.boxplot([g, b], showfliers=False); a.set_xticklabels(["好", "坏"])
        for j, arr in enumerate([g, b]): a.scatter(np.random.normal(j + 1, .05, len(arr)), arr, s=10, alpha=.6)
        a.set_title(f"{FEATURE_NAME[f]}\nAUC={name_by[f]['auc']:.2f} TV={name_by[f]['TV']:.2f}", fontsize=9); a.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_g2_box.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    fig, a = plt.subplots(figsize=(5.6, 5.2))
    for f in top[:4]:
        fpr, tpr, av = roc[f]; a.plot(fpr, tpr, label=f"{FEATURE_NAME[f]} ({av:.2f})")
    a.plot([0, 1], [0, 1], "--", color="#999"); a.set_xlabel("假好率 FPR"); a.set_ylabel("真好率 TPR")
    a.set_title("单标量 ROC（越近对角线=越必败）"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_g2_roc.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    # TV/KS 条形（命题 ii：单标量边缘近重合 ⇒ TV/KS 小）
    order = sorted(table, key=lambda t: t["feature"])
    fig, a = plt.subplots(figsize=(9, 4.4)); xs = np.arange(len(order)); w = .4
    a.bar(xs - w / 2, [t["TV"] for t in order], w, label="TV", color="#8BC8EA")
    a.bar(xs + w / 2, [t["KS"] for t in order], w, label="KS", color="#E1B98F")
    a.axhline(.2, color="#c0392b", ls="--", lw=1, label="τ=0.2 近重合参考")
    a.set_xticks(xs); a.set_xticklabels([FEATURE_NAME[t["feature"]] for t in order], rotation=40, ha="right", fontsize=8)
    a.set_ylabel("好/坏边缘分布距离"); a.set_title("命题8.8(ii)：单标量 TV/KS 越小越无法判别"); a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(OUT, "fig_g2_tvks.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    # 分组留一对比（命题 iii：结构/多变量显著高于幅值单标量）
    if groups:
        fig, a = plt.subplots(figsize=(7.4, 4.4)); xs = np.arange(len(groups)); w = .25
        for k, c in enumerate(["loo_sens", "loo_spec", "loo_acc"]):
            a.bar(xs + (k - 1) * w, [g[c] for g in groups], w, label={"loo_sens": "灵敏度", "loo_spec": "特异度", "loo_acc": "准确率"}[c])
        for i, g in enumerate(groups): a.text(i, max(g["loo_acc"], g["loo_sens"]) + .02, f"AUC={g['auc']:.2f}", ha="center", fontsize=9)
        a.set_xticks(xs); a.set_xticklabels([g["group"] for g in groups]); a.set_ylim(0, 1.08)
        a.set_ylabel("留一交叉验证"); a.set_title("命题8.8(iii)：多变量结构量 vs 幅值单标量组"); a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
        fig.tight_layout(); p = os.path.join(OUT, "fig_g2_groups.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    if len(picks) >= 2:
        fig, a = plt.subplots(figsize=(6.4, 5.4))
        for cls, cc, lab in [(1, "#3a8a18", "好盆地"), (0, "#c0392b", "坏盆地")]:
            z = [r for r in rows if r["truth_good"] == cls]
            a.scatter([r[picks[0]] for r in z], [r[picks[1]] for r in z], c=cc, label=lab, s=22, alpha=.7)
        a.set_xlabel(FEATURE_NAME[picks[0]]); a.set_ylabel(FEATURE_NAME[picks[1]])
        a.set_title("top2 无真解特征平面"); a.legend(); a.grid(alpha=.3)
        fig.tight_layout(); p = os.path.join(OUT, "fig_g2_scatter.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


def main():
    rows, rows_oracle = collect()
    toy = toy_fourpoint()
    if len({r["truth_good"] for r in rows}) < 2:
        print("[G2] 主口径好/坏两类未同时出现，无法判别；已落样本，扩样后重跑。"); return
    _, table, roc = analyze(rows)
    groups = group_loo(rows)
    combo, picks, dirs, thr = combo_loo(rows, table, top=2)
    pseudo = pseudo_report(rows, picks, dirs, thr)
    # oracle 上界（truth 选起点）
    if len({r["truth_good"] for r in rows_oracle}) == 2:
        _, table_o, _ = analyze(rows_oracle, tag="_oracle")
        groups_o = group_loo(rows_oracle, tag="_oracle")
    else:
        table_o, groups_o = [], []
    plots(rows, table, roc, picks, groups)
    print("\n===== G2 主口径（proxy 无真解选起点）单标量判别（留一）=====")
    print("特征              组     AUC   TV     KS    留一sens spec acc")
    for t in table:
        print(f"  {t['feature']:<12}{t['group']:<6}{t['auc']:.2f}  {t['TV']:.2f}  {t['KS']:.2f}  "
              f"{t['loo_sens']:.2f}  {t['loo_spec']:.2f}  {t['loo_acc']:.2f}")
    print("\n多变量分组留一：")
    for g in groups: print(f"  {g['group']:<14} nfeat={g['n_feat']:>2} AUC={g['auc']:.2f} "
                           f"sens={g['loo_sens']:.2f} spec={g['loo_spec']:.2f} acc={g['loo_acc']:.2f}")
    print("\n四点坐标例自检（注记8.10）：")
    for z in toy: print(f"  {z['probe']:<32} AUC={z['auc']:.3f}")
    if groups_o:
        print("\noracle 上界（truth 选起点，仅参考）：")
        for g in groups_o: print(f"  {g['group']:<14} AUC={g['auc']:.2f} acc={g['loo_acc']:.2f}")
    print(f"\n伪好/伪坏 = {len(pseudo)}；选起点一致率(select_agree)="
          f"{np.mean([r['select_agree'] for r in rows]):.2f}（主/上界差距即选起点损失）")
    print("[G2] 完成，产物在", OUT)


if __name__ == "__main__":
    main()
