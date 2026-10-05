# -*- coding: utf-8 -*-
"""
实验 E1b（V10.2）：一维稳态 Burgers 的【大 ν 光滑退化端】，补齐 β*(ν) 的左端平台
--------------------------------------------------------------------------------
科学问题：E1 的 ν 只到 .01（激波端），只证了“越难 β* 越大”，没证“足够光滑 → β*→1、
加权退回无用”。本实验把 ν 向大（光滑）端扩，并隔离一个关键混杂：现网络为薄激波用了
高频傅里叶嵌入（σ=15），大 ν 解光滑近线性，沿用高频嵌入会造成“表示器谱带宽—解频谱失配”
的伪困难。故用 2 因素设计：ν（光滑度）× 表示器（三档）：

  feat='Hi'    : 傅里叶、固定高频 σ=15（现状，对照组；大 ν 若仍显 β 增益=表示失配证据）
  feat='Match' : 傅里叶、带宽匹配 σ(ν)=SIG_REF*NU_REF/ν=.075/ν（σ(.005)=15 与现状锚定，预注册不调）
  feat='MLP'   : 关闭傅里叶映射、直接以 x 输入（最干净的 null 主干，零额外超参）

预注册预期（跑完只做 坐实/降格/查反例）：
  ν≥.05 后，MLP 与 Match：β* 中位=1、L2-β 平坦（gain≈0）、η_gate-β 的 Spearman→0；
  且激波带占比 frac_S→1、对比度间隙 Δ 与越序占比→0（C-Sep 分区在光滑端平凡化、机制 A 无对象）。
  Hi 若在大 ν 残留 β 增益，归因“表示器—解频谱失配”，与谱偏差/NTK 文献呼应（非物理失配）。

设计：仍走“Phase0 进盆地缓存 → 同一盆地上扫 β（唯一变量 β）”，与 E1 同口径；
     三档仅改表示器/带宽，训练内循环、权函数族(rational)、指标全部复用公共层，不改 r2s/mc。

V10.2 三项工程化（省算力 + 抗中断 + 预注册分层，不挑种子）：
  (1) 历史复用（严格同条件才复用，逐 run 标 source 可追溯）：
      - Hi 档(fourier,σ=15,rational,8000,中心架构64/4/1e4) 与主线 E1(exp_E1_beta_nu) 是【同一计算】：
        ν=.005/.01 上、β 网格交集部分直接导入 e1_runs.csv，不重训；
      - R1(exp_R1_arch) 中 (hidden,layers,n_pde)=(64,4,10000)、family=rational 的基准档同为 σ15/fourier，
        ν=.1、β∈{1,3,8}、SEEDS_NEW 直接导入（E1B_USE_R1=0 可关）；
      - Match 仅在 ν=.005 时 σ_match=15 与 Hi 物理等价，直接共享 Hi/.005 结果，不重复训练；
      - MLP 档(关傅里叶)历史无可复用，全部新训；β 必须精确相等，E1 的 7/10/15 绝不顶替 8/12。
  (2) 两阶段种子（等距严格超集，实验前固定）：
      - 激波端 ν∈{.005,.01,.03} 始终用 SEEDS_EXT(11)；
      - 光滑端 ν≥.05：阶段一(E1B_STAGE=1,默认)先 SEEDS_NEW(5)；阶段二(E1B_STAGE=2)扩到 SEEDS_EXT(11)，
        SEEDS_NEW⊂SEEDS_EXT，断点机制自动只补新 6 个种子，不重跑。
  (3) 断点续跑：启动先读已有 e1b_runs/e1b_csep，已完成键跳过；每跑完 1 个 run 立即落盘，
      Windows 更新重启之类的中断只丢当前 1 个 run。

产出 results/v6/exp_E1b_smooth/：
  e1b_runs.csv（逐 run，含 source 列）/ e1b_csep.csv / e1b_summary.csv
  fig_E1b_beta_star.png / fig_E1b_L2_flat.png / fig_E1b_csep.png / fig_E1b_eta_spearman.png
SMOKE：V8_SMOKE=1
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc

for _fp in (r"C:\Windows\Fonts\msyh.ttc",):
    if os.path.exists(_fp):
        try: font_manager.fontManager.addfont(_fp)
        except Exception: pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = str(mc.RES_ROOT / "exp_E1b_smooth"); os.makedirs(OUT, exist_ok=True)
r2s.OUT = OUT
RES = mc.RES_ROOT
E1_CSV = RES / "exp_E1_beta_nu" / "e1_runs.csv"
R1_CSV = RES / "exp_R1_arch" / "r1_runs.csv"
RUNS_CSV = os.path.join(OUT, "e1b_runs.csv")
CSEP_CSV = os.path.join(OUT, "e1b_csep.csv")

# ---------------- 实验网格（实验前固定，不挑点） ----------------
NU_REF, SIG_REF = 0.005, 15.0            # σ(ν) 锚点：现状 σ=15 对应 ν=.005
def sigma_match(nu): return SIG_REF * NU_REF / float(nu)     # =.075/ν，频率∝1/层厚(2ν)
FEATS = ["MLP", "Match", "Hi"]           # 输出/绘图顺序
FEATS_RUN = ["Hi", "Match", "MLP"]       # 调度顺序：保证 Match/.005 能共享已就绪的 Hi/.005
FEAT_COLOR = {"MLP": "#E8A24B", "Match": "#4B86B4", "Hi": "#9A9A9A"}
NUS = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
SHOCK_NU = {0.005, 0.01, 0.03}
BETAS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]  # 光滑端无需大 β，看是否平坦
BAND_C = 3.0                             # 激波带 |x|<3ν
STAGE = os.environ.get("E1B_STAGE", "1") # 1=光滑端5seed；2=光滑端补到11seed
USE_R1 = os.environ.get("E1B_USE_R1", "1") != "0"
if mc.SMOKE:
    NUS = [0.1]; BETAS = [1.0, 5.0]
P1_EPOCHS = 200 if mc.SMOKE else 8000    # 统一预算上限（与主线一致，堵“光滑端步数少→假退化”）；实际收敛点由 L2 平台判据记录
PROBE_N = 256 if mc.SMOKE else 2048
CONV_WIN, CONV_TOL = 5, 1e-3             # 平台判据：连续 5 个评估点 L2 相对极差 <1e-3 视为进入平台

RUN_COLS = ["feat", "nu", "seed", "beta", "phase0_L2", "gate_ok", "L2", "Linf", "l2_band",
            "best_epoch", "conv_epoch", "converged", "eta_gate", "eta_end", "cos_gate", "source"]

def seeds_for_nu(nu):
    """激波端恒 11 seed；光滑端按 STAGE 取 5 或 11（5⊂11 等距超集）。SMOKE 单种子。"""
    if mc.SMOKE: return [mc.SEEDS_EXT[0]]
    if float(nu) in SHOCK_NU: return list(mc.SEEDS_EXT)
    return list(mc.SEEDS_EXT if STAGE == "2" else mc.SEEDS_NEW)

def _f(x, d=float):
    try: return d(x)
    except Exception: return np.nan

def rkey(feat, nu, seed, beta):
    return (feat, round(float(nu), 6), int(seed), round(float(beta), 6))

def conv_epoch(dyn, win=CONV_WIN, tol=CONV_TOL):
    """从验证 L2 轨迹识别最早进入平台的 epoch；整段未平台化则返回末点并置 converged=False。"""
    es = np.asarray(dyn.get("eval_step", []), float); l2 = np.asarray(dyn.get("L2_traj", []), float)
    if l2.size < win + 1:
        return (int(es[-1]) if es.size else -1, False)
    for i in range(0, l2.size - win + 1):
        w = l2[i:i + win]; denom = abs(np.median(w)) + 1e-12
        if (np.max(w) - np.min(w)) / denom < tol:
            return int(es[i]), True
    return int(es[-1]), False

# ---- 进程内包装 r2s.base_cfg（不改公共文件）：①所有档注入正确硬边界幅值 bc_amp=tanh(1/2ν)，
#     使端点 u(±1)=∓tanh(1/2ν) 与真解一致；②MLP 档额外注入 feature='mlp'。小 ν tanh≈1，与旧 A=-x 等价 ----
_orig_base_cfg = r2s.base_cfg
def _bc_base_cfg(nu, seed_tag):
    c = _orig_base_cfg(nu, seed_tag)
    c["bc_amp"] = float(np.tanh(1.0 / (2.0 * float(nu))))
    return c
def _mlp_base_cfg(nu, seed_tag):
    c = _bc_base_cfg(nu, seed_tag); c["feature"] = "mlp"; return c
def set_feat(feat):
    r2s.base_cfg = _mlp_base_cfg if feat == "MLP" else _bc_base_cfg
r2s.base_cfg = _bc_base_cfg       # 启动即切到正确边界，避免遗漏路径仍用旧 A=-x

def feat_sigma(feat, nu):
    if feat == "Hi": return 15.0
    if feat == "Match": return sigma_match(nu)
    return 1.0                              # MLP 无傅里叶，占位仅用于缓存 tag 区分

# ---------------- C-Sep 平凡化探针（从冻结对比度场 t_field 现算） ----------------
def csep_from_basin(basin, nu):
    t = basin["t_field"].detach().cpu().numpy().reshape(-1)
    x = np.linspace(-1.0, 1.0, t.size)
    msk = np.abs(x) < BAND_C * float(nu)
    frac_s = float(msk.mean())
    out = dict(frac_S=frac_s, minS=np.nan, maxO=np.nan, Delta=np.nan,
               meanS=np.nan, meanO=np.nan, inversion=np.nan)
    if msk.any() and float((~msk).mean()) >= 0.50:  # O区占比需≥50%(frac_S≤.5)才具统计代表性; frac_S>.5(ν≳1/6)时 S/O 失去激波/光滑语义, C-Sep 标 N/A
        tS, tO = t[msk], t[~msk]
        minS, maxO = float(tS.min()), float(tO.max())
        out.update(minS=minS, maxO=maxO, Delta=minS - maxO,
                   meanS=float(tS.mean()), meanO=float(tO.mean()),
                   inversion=float(np.mean(tO > minS)))   # O 中越过 S 最小值的比例（严格 C-Sep 违反度）
    return out

def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 3 and np.unique(x[m]).size >= 2 and np.unique(y[m]).size >= 2:
            res = spearmanr(x[m], y[m])
            return float(res.statistic if hasattr(res, "statistic") else res[0])
    except Exception:
        pass
    return np.nan

# ---------------- 断点续跑：读已有结果 ----------------
def _read_csv_safe(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_existing_runs():
    """已有 e1b_runs.csv → (rows, done)；数值列规范化。"""
    rows, done = [], set()
    for d in _read_csv_safe(RUNS_CSV):
        try:
            r = dict(feat=d["feat"], nu=float(d["nu"]), seed=int(float(d["seed"])),
                     beta=float(d["beta"]), source=d.get("source", "resume"))
            for k in ("phase0_L2", "gate_ok", "L2", "Linf", "l2_band", "best_epoch",
                      "conv_epoch", "converged", "eta_gate", "eta_end", "cos_gate"):
                r[k] = _f(d.get(k, ""))
            rows.append(r); done.add(rkey(r["feat"], r["nu"], r["seed"], r["beta"]))
        except Exception:
            continue
    return rows, done

def load_existing_csep():
    out, done = [], set()
    for d in _read_csv_safe(CSEP_CSV):
        try:
            r = {k: _f(v) for k, v in d.items() if k not in ("feat",)}
            r["feat"] = d["feat"]; r["nu"] = float(d["nu"]); r["seed"] = int(float(d["seed"]))
            out.append(r); done.add((r["feat"], round(r["nu"], 6), r["seed"]))
        except Exception:
            continue
    return out, done

# ---------------- 历史复用：严格同条件导入 ----------------
def load_historical(done):
    """从 E1/R1 导入与 Hi 档严格同条件的历史 run；返回新增 rows 列表（不覆盖 done）。"""
    add = []
    def take(feat, nu, seed, beta, src, d, mapping):
        k = rkey(feat, nu, seed, beta)
        if k in done: return
        r = dict(feat=feat, nu=float(nu), seed=int(seed), beta=float(beta),
                 phase0_L2=np.nan, gate_ok=np.nan, L2=np.nan, Linf=np.nan, l2_band=np.nan,
                 best_epoch=np.nan, conv_epoch=-1, converged=-1,
                 eta_gate=np.nan, eta_end=np.nan, cos_gate=np.nan, source=src)
        for rk, sk in mapping.items():
            r[rk] = _f(d.get(sk, ""))
        add.append(r); done.add(k)

    # E1：fourier σ15 rational 8000 中心架构 == Hi 档；ν=.005/.01，β 取交集
    if os.path.exists(E1_CSV):
        for d in _read_csv_safe(E1_CSV):
            nu = float(d["nu"]); beta = float(d["beta"]); seed = int(float(d["seed"]))
            if nu not in (0.005, 0.01) or not any(abs(beta - b) < 1e-9 for b in BETAS): continue
            mp = dict(phase0_L2="phase0_L2", gate_ok="gate_ok", L2="L2", Linf="Linf",
                      l2_band="l2_band", best_epoch="best_epoch",
                      eta_gate="eta_gate", eta_end="eta_end", cos_gate="cos_gate")
            take("Hi", nu, seed, beta, "hist_E1", d, mp)

    # R1：仅基准档 (64,4,10000)+rational+fourier σ15 == Hi 档；ν=.1，β∈{1,3,8}，SEEDS_NEW
    if USE_R1 and os.path.exists(R1_CSV):
        for d in _read_csv_safe(R1_CSV):
            try:
                if d.get("family") != "rational": continue
                if (int(float(d["hidden"])), int(float(d["layers"])), int(float(d["n_pde"]))) != (64, 4, 10000): continue
                nu = float(d["nu"]); beta = float(d["beta"]); seed = int(float(d["seed"]))
                continue  # 停用 R1/.1 导入：ν=.1 旧档为 A=-x，修复后改由正确边界重训做新旧并排
                if abs(nu - 0.1) > 1e-9 or not any(abs(beta - b) < 1e-9 for b in (1.0, 3.0, 8.0)): continue
                if seed not in mc.SEEDS_NEW: continue
                mp = dict(phase0_L2="phase0_L2", L2="L2", eta_gate="eta_gate")
                take("Hi", nu, seed, beta, "hist_R1", d, mp)
            except Exception:
                continue
    return add

# ---------------- 主采集 ----------------
def collect():
    rows, done = load_existing_runs()
    csep_rows, done_csep = load_existing_csep()
    n0 = len(rows)
    hist = load_historical(done)
    rows.extend(hist)
    n_hist = len(hist)
    n_train = n_share = 0

    def _save():
        mc.write_csv(RUNS_CSV, [ {k: r.get(k, "") for k in RUN_COLS} for r in rows ])
        mc.write_csv(CSEP_CSV, csep_rows)

    for feat in FEATS_RUN:
        set_feat(feat)
        for nu in NUS:
            xt, ut = r2s.build_test(nu)
            xp = torch.linspace(-1, 1, PROBE_N, device=mc.DEVICE).view(-1, 1)
            sig = feat_sigma(feat, nu)
            seeds = seeds_for_nu(nu)
            for seed in seeds:
                # C-Sep（走盆地缓存，命中零训练；断点去重）
                if (feat, round(float(nu), 6), int(seed)) not in done_csep:
                    basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=sig,
                                                  k=4, phase0_epochs=2000)
                    csep_rows.append(dict(feat=feat, nu=nu, seed=seed, **csep_from_basin(basin, nu)))
                    done_csep.add((feat, round(float(nu), 6), int(seed)))
                else:
                    basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=sig,
                                                  k=4, phase0_epochs=2000)
                for beta in BETAS:
                    k = rkey(feat, nu, seed, beta)
                    if k in done: continue
                    # Match/.005 与 Hi/.005 物理等价（σ 都=15、均 fourier）：共享，不重训
                    if feat == "Match" and abs(float(nu) - NU_REF) < 1e-12:
                        src = next((r for r in rows
                                    if rkey("Hi", nu, seed, beta) == rkey(r["feat"], r["nu"], r["seed"], r["beta"])), None)
                        if src is not None:
                            r = dict(src); r["feat"] = "Match"
                            r["source"] = "shared_Hi(nu=.005)|" + str(src.get("source", ""))
                            rows.append(r); done.add(rkey("Match", nu, seed, beta)); n_share += 1
                            _save()
                            print(f"[E1b Match] ν{nu} s{seed} β{beta:g}: 共享 Hi 结果 L2={r['L2']:.3e}")
                            continue
                    m, dyn, model, crit, tr = mc.train_phase1_family(
                        basin, nu, beta, seed, xt, ut, family="rational",
                        epochs=P1_EPOCHS, tag=f"E1b_{feat}", grad_probe=(xp, nu))
                    gp = tr.grad_probe
                    ce, isconv = conv_epoch(dyn)
                    r = dict(feat=feat, nu=float(nu), seed=int(seed), beta=float(beta),
                             phase0_L2=basin["l2"], gate_ok=int(basin["l2"] < mc.GATE_L2_MAX),
                             L2=m.get("L2_Error", np.nan), Linf=m.get("L_inf_Error", np.nan),
                             l2_band=m.get("l2_band", np.nan),
                             best_epoch=int(getattr(tr, "best_epoch", -1)),
                             conv_epoch=ce, converged=int(isconv),
                             eta_gate=(gp["gate"] or {}).get("eta", np.nan),
                             eta_end=(gp["end"] or {}).get("eta", np.nan),
                             cos_gate=(gp["gate"] or {}).get("cos", np.nan), source="new")
                    rows.append(r); done.add(k); n_train += 1
                    _save()
                    print(f"[E1b {feat}] ν{nu} s{seed} β{beta:g}: L2={r['L2']:.3e} η_g={r['eta_gate']:.2f}")
    set_feat("Hi")  # 复原
    _save()
    print(f"\n[E1b] 启动前已有 {n0}；历史导入 {n_hist}；本次新训 {n_train}；Match/.005 共享 {n_share}；"
          f"当前合计 {len(rows)}（STAGE={STAGE}）")
    return rows, csep_rows

# ---------------- 汇总：β*、β 增益、η-β Spearman ----------------
def summarize(rows):
    summ = []
    for feat in FEATS:
        for nu in NUS:
            sub = [r for r in rows if r["feat"] == feat and r["nu"] == nu]
            if not sub: continue
            bstars, gains = [], []
            spear = []
            for seed in sorted({r["seed"] for r in sub}):
                z = sorted([(r["beta"], r["L2"]) for r in sub if r["seed"] == seed])
                zb = [(b, v) for b, v in z if np.isfinite(v)]
                if not zb: continue
                l2_1 = next((v for b, v in zb if abs(b - 1.0) < 1e-9), np.nan)
                bbest, vbest = min(zb, key=lambda t: t[1])
                bstars.append(bbest)
                if np.isfinite(l2_1) and vbest > 0: gains.append(l2_1 / vbest - 1.0)
                sp = _spearman([b for b, _ in z], [r["eta_gate"] for r in
                               [rr for rr in sub if rr["seed"] == seed]])
                if np.isfinite(sp): spear.append(sp)
            l2all = np.asarray([r["L2"] for r in sub], float)
            known = [r for r in sub if r.get("converged", -1) in (0, 1)]  # 历史行 converged=-1 未知，不计收敛率
            summ.append(dict(feat=feat, nu=nu, n=len(bstars),
                             beta_star_med=np.median(bstars) if bstars else np.nan,
                             beta_star_min=np.min(bstars) if bstars else np.nan,
                             beta_star_max=np.max(bstars) if bstars else np.nan,
                             gain_med=np.median(gains) if gains else np.nan,
                             gain_q1=np.quantile(gains, .25) if gains else np.nan,
                             gain_q3=np.quantile(gains, .75) if gains else np.nan,
                             eta_beta_spearman=np.median(spear) if spear else np.nan,
                             eta_applicable=int(3.0*nu < 1.0-1e-12),  # 3ν>=1 时无带外光滑区、η 不适用(N/A)
                             conv_epoch_med=np.median([r["conv_epoch"] for r in known
                                                       if r.get("conv_epoch", -1) >= 0]) if known else np.nan,
                             frac_converged=np.mean([r.get("converged", 0) for r in known]) if known else np.nan,
                             L2_med=np.nanmedian(l2all)))
    mc.write_csv(os.path.join(OUT, "e1b_summary.csv"), summ)
    return summ

# ---------------- 图 ----------------
def _med_iqr(sub, key, xs):
    med, lo, hi = [], [], []
    for x in xs:
        v = np.asarray([r[key] for r in sub if r["beta"] == x], float); v = v[np.isfinite(v)]
        med.append(np.median(v) if v.size else np.nan)
        lo.append(np.quantile(v, .25) if v.size else np.nan)
        hi.append(np.quantile(v, .75) if v.size else np.nan)
    return np.asarray(med), np.asarray(lo), np.asarray(hi)

def plot_beta_star(summ):
    fig, a = plt.subplots(figsize=(6.6, 4.6))
    for feat in FEATS:
        d = [s for s in summ if s["feat"] == feat]
        xs = [s["nu"] for s in d]; ym = [s["beta_star_med"] for s in d]
        lo = [s["beta_star_min"] for s in d]; hi = [s["beta_star_max"] for s in d]
        a.plot(xs, ym, "o-", color=FEAT_COLOR[feat], label=feat)
        a.fill_between(xs, lo, hi, alpha=.15, color=FEAT_COLOR[feat])
    a.axhline(1.0, color="k", ls=":", lw=1, label="β*=1（退化）")
    a.set_xscale("log"); a.set_xlim(min(NUS) * 0.8, max(NUS) * 1.25)
    a.set_xlabel("ν（越大解越光滑）"); a.set_ylabel("β*（跨种子中位/极差）")
    a.set_title("β*(ν) 完整曲线：大 ν 光滑端应收敛到 β*=1"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1b_beta_star.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_L2_flat(rows):
    pick = [nu for nu in [0.01, 0.05, 0.1, 0.5] if nu in NUS][:4]
    if not pick: return
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), squeeze=False)
    for a, nu in zip(ax.ravel(), pick):
        for feat in FEATS:
            sub = [r for r in rows if r["feat"] == feat and r["nu"] == nu]
            if not sub: continue
            med, lo, hi = _med_iqr(sub, "L2", BETAS)
            a.plot(BETAS, med, "o-", color=FEAT_COLOR[feat], label=feat)
            a.fill_between(BETAS, lo, hi, alpha=.12, color=FEAT_COLOR[feat])
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"ν={nu}：L2 对 β（平坦=加权无增益）")
        a.set_xlabel("β"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1b_L2_flat.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_csep(csep_rows):
    fig, a1 = plt.subplots(figsize=(6.8, 4.6))
    xall = sorted({r["nu"] for r in csep_rows})
    # frac_S 与 ν 的关系对表示器不敏感，用 MLP 主档；Δ/越序三档淡线
    base = [r for r in csep_rows if r["feat"] == "MLP"]
    xs = [r["nu"] for r in base]
    a1.plot(xs, [r["frac_S"] for r in base], "k--", label="激波带占比 frac_S")
    a1.plot(xs, [r["inversion"] for r in base], "o-", color="#E8A24B", label="越序占比（MLP）")
    for feat in ("Match", "Hi"):
        d = [r for r in csep_rows if r["feat"] == feat]
        a1.plot([r["nu"] for r in d], [r["inversion"] for r in d], ":",
                color=FEAT_COLOR[feat], alpha=.7, label=f"越序占比（{feat}）")
    a1.set_xscale("log"); a1.set_xlim(min(xall) * 0.8, max(xall) * 1.25)
    a1.set_xlabel("ν"); a1.set_ylim(-.05, 1.05)
    a1.set_ylabel("frac_S / 越序占比"); a1.grid(alpha=.3); a1.legend(fontsize=8)
    a1.set_title("C-Sep 平凡化：光滑端 S 带吞并全域、对比度越序消失")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1b_csep.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_eta_spearman(summ):
    fig, a = plt.subplots(figsize=(6.6, 4.4))
    for feat in FEATS:
        d = [s for s in summ if s["feat"] == feat]
        a.plot([s["nu"] for s in d], [s["eta_beta_spearman"] for s in d],
               "o-", color=FEAT_COLOR[feat], label=feat)
    a.axhline(0, color="k", lw=1); a.set_xscale("log")
    a.set_xlim(min(NUS) * 0.8, max(NUS) * 1.25); a.set_ylim(-1.05, 1.05)
    a.set_xlabel("ν"); a.set_ylabel("Spearman(β, η_gate)")
    a.set_title("机制 A 方向中性：光滑端 η 不再随 β 单调（→0）"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1b_eta_spearman.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plan_preview():
    """不开训，打印本阶段 复用/共享/新训 账，便于开跑前核对。"""
    rows, done = load_existing_runs()
    hist = load_historical(done); rows.extend(hist)
    n_new = n_share = 0
    for feat in FEATS_RUN:
        for nu in NUS:
            for seed in seeds_for_nu(nu):
                for beta in BETAS:
                    if rkey(feat, nu, seed, beta) in done: continue
                    if feat == "Match" and abs(float(nu) - NU_REF) < 1e-12: n_share += 1
                    else: n_new += 1
    src = {}
    for r in rows: src[r.get("source", "?")] = src.get(r.get("source", "?"), 0) + 1
    print(f"[E1b plan STAGE={STAGE}] 已就位(历史/断点)={len(rows)} {src}；Match/.005 待共享={n_share}；需新训={n_new}")

def main():
    if os.environ.get("E1B_PLAN") == "1":
        plan_preview(); return
    rows, csep_rows = collect()
    summ = summarize(rows)
    plot_beta_star(summ); plot_L2_flat(rows); plot_csep(csep_rows); plot_eta_spearman(summ)
    print("\n[E1b] ν  feat   β*med  gain_med  Spearman_ηβ")
    for s in summ:
        print(f"  {s['nu']:<6} {s['feat']:<5} {s['beta_star_med']:>4g}  "
              f"{s['gain_med']:+.3f}   {s['eta_beta_spearman']:+.2f}")
    print("\n[E1b] 完成。产物在", OUT)

if __name__ == "__main__":
    main()
