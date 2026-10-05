# -*- coding: utf-8 -*-
"""
V2 实验 S2c：同配置两阶段下 Burgers 激波 与 定常对流-扩散边界层 的“厚度对齐”对照
================================================================================
动机（修正单阶段 S2b 的分辨限误判）：
  单阶段 S2b 在 ε≤0.01 全体失败，但同厚度 Burgers（ν=.005↔ε=.01，内尺度 2ν=ε）在
  【σ 表示课程 5→15 + multistart K=4 + Gate + 两阶段 + best-ckpt】下能解到全域 L2≈8e-3（R2）。
  本脚本把 Burgers 验证过的【完整配置原样迁移】到对流-扩散：
    * Burgers 侧直接调用权威的 mc.enter_basin_cached / mc.train_phase1_family（与 R2 同一函数、
      同一盆地缓存、bc_amp=tanh(1/2ν) 补丁），数字即 R2 同条件金标准；
    * 对流-扩散(pos)侧在本脚本内【逐行复刻】 r2s.run_phase0 / mc.train_phase1_family 的装配，
      仅做三处物理上必须的替换：PDE=ConvDiffPDE1D、边界提升=lift_linear(0→1)、内层掩膜由
      居中 |x|<3ν 改为贴右边界 x≥1−5ε（LayerTrainer 只覆盖 _band_l2，其余训练逻辑不动）。
  从而在“同配置”下检验边界层能否复现 Burgers 同厚度精度（β=1 baseline），再在同一好盆地上
  扫族-β（全量），检验局域刚性迁移后空间加权是否再激活。

厚度对应（10–90 过渡全宽相等；内尺度也一致）：ε_eq = 2 ν。
  ν=.005↔ε=.01，ν=.001↔ε=.002，ν=.003↔ε=.006，ν=.01↔ε=.02。

不修改任何公共代码。盆地缓存：bur 走 mc 共享缓存 results/v6/_gate_cache；pos 走本目录 _cache。
运行：
  set V8_SMOKE=1 && python exp_V2_S2c_aligned_twostage.py --baseline
  python exp_V2_S2c_aligned_twostage.py --baseline
  python exp_V2_S2c_aligned_twostage.py --cases pos:0.01 --seeds 0,5,10 \
      --families uniform:1,rational:1,rational:3,rational:10,down_inv:1,down_inv:3,down_inv:10
================================================================================
"""
import os, sys, csv, argparse, copy
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from physics.burgers_pde import SteadyBurgersPDE
from physics.convdiff_pde import ConvDiffPDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.family_weighting_loss import FamilyWeightingLoss
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2c_aligned_twostage")
CACHE = os.path.join(OUT, "_cache"); os.makedirs(CACHE, exist_ok=True)
r2s.OUT = OUT                       # 训练日志写到本实验目录
DEV = mc.DEVICE

# ---- 与 R2/Burgers 完全一致的配置（pos 侧逐行复刻，不另设超参）----
N_TEST = 2048
FIELDS = ["kind", "delta", "equiv_nu", "seed", "family", "beta", "p0", "p1",
          "phase0_L2", "L2", "Linf", "L2_layer", "L2_outer", "gate_epoch", "best_epoch", "k"]

# 正确硬边界端点幅值（与 exp_R2_budget_converge.py 同一进程内补丁；小 ν≈1，lift_linear 不读此项）
_orig_base_cfg = r2s.base_cfg
def _bc_base_cfg(nu, seed_tag):
    c = _orig_base_cfg(nu, seed_tag)
    c["bc_amp"] = float(np.tanh(1.0 / (2.0 * float(nu))))
    return c
r2s.base_cfg = _bc_base_cfg


class LayerTrainer(PINNTrainer):
    """仅把内层掩膜泛化为贴右边界（pos）；center 与父类逐字节相同。best-ckpt/Gate/σ课程/两阶段全沿用。
    pos 把 cfg['nu']=ε、shock_band_c=5，则 Gate 峰值容差 5ε 与内层宽 5ε 自动正确。"""
    def __init__(self, *a, layer_side="center", **kw):
        super().__init__(*a, **kw)
        self.layer_side = str(layer_side)

    def _layer_mask(self, xn, delta=None, c=None):
        band = (c if c is not None else self.shock_band_c) * (delta if delta is not None else self.nu)
        return (xn >= 1.0 - band) if self.layer_side == "right" else (np.abs(xn) < band)

    def _band_l2(self, test_data):
        self.model.eval()
        with torch.no_grad():
            up = self.model(test_data["x"]).detach().cpu().numpy().flatten()
        xn = test_data["x"].detach().cpu().numpy().flatten()
        ut = np.asarray(test_data["u_true"]).flatten()
        m = self._layer_mask(xn)
        self.model.train()
        return float(np.linalg.norm((up - ut)[m]) / max(np.linalg.norm(ut[m]), 1e-8)) if m.any() else 0.0


# ----------------------------- case 定义 -----------------------------
def pos_exact(eps):
    pde = ConvDiffPDE1D(dict(cd_eps=eps, cd_b=1.0, bc_left=0.0, bc_right=1.0))
    return pde, (lambda x, _p=pde: _p.exact_np(x))


def pos_base_cfg(eps, tag):
    c = r2s.base_cfg(eps, tag)           # nu=ε，供 trainer 的 band/Gate 尺度使用
    c.update(dict(shock_band_c=5.0, bc_type="lift_linear", bc_left=0.0, bc_right=1.0,
                  cd_eps=eps, cd_b=1.0, layer_side="right"))
    return c


def test_tensors(kind, delta):
    xn = np.linspace(-1, 1, N_TEST, dtype=np.float64).reshape(-1, 1)
    if kind == "bur":
        un = -np.tanh(xn / (2.0 * delta)).astype(np.float64)
    else:
        _, ex = pos_exact(delta); un = ex(xn).astype(np.float64)
    xt = torch.tensor(xn, dtype=torch.float32, device=DEV)
    return xt, un


def quick_l2(model, xt, un):
    model.eval()
    with torch.no_grad():
        up = model(xt).detach().cpu().numpy().astype(np.float64).flatten()
    model.train()
    return float(np.linalg.norm(up - un.flatten()) / max(np.linalg.norm(un), 1e-12))


def final_eval_pos(model, xt, un, eps):
    model.eval()
    with torch.no_grad():
        pred = model(xt).detach().cpu().numpy().astype(np.float64).flatten()
    x = xt.detach().cpu().numpy().flatten(); u = un.flatten(); d = pred - u
    m = x >= 1.0 - 5.0 * eps
    g = max(np.linalg.norm(u), 1e-12)
    model.train()
    return dict(L2=float(np.linalg.norm(d) / g), Linf=float(np.max(np.abs(d))),
                L2_layer=float(np.linalg.norm(d[m]) / max(np.linalg.norm(u[m]), 1e-12)),
                L2_outer=float(np.linalg.norm(d[~m]) / g))


# --------------------- pos：逐行复刻 r2s.run_phase0 ---------------------
def enter_basin_pos(eps, base_seed, xt, un, k, p0, reuse=True, sigma_hi=15.0, return_all=False):
    tag = f"pos{eps:g}_s{base_seed}_sig{sigma_hi:g}_k{k}_p0{p0}.pt"
    path = os.path.join(CACHE, tag)
    r2s.MULTISTART_K = 1 if mc.SMOKE else int(k)
    r2s.SELECT_BY = "truth"; r2s.SIG_HI = float(sigma_hi); r2s.SIGMA_HI_CFG = float(sigma_hi)
    r2s.PHASE0_EPOCHS = 60 if mc.SMOKE else int(p0)
    if reuse and os.path.exists(path):
        d = torch.load(path, map_location=DEV)
        print(f"[pos Cache] hit {tag} Phase0_L2={d['l2']:.3e}")
        return (d, None) if return_all else d
    td = {"x": xt, "u_true": un.flatten()}
    cands = []
    for kk in range(r2s.MULTISTART_K):
        seed = base_seed + kk              # 与 r2s 一致：multistart 内部间隔 1
        r2s.set_seed(seed)
        cfg = pos_base_cfg(eps, f"p0_pos{eps:g}_{base_seed}_{kk}")
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG, sigma_anneal_T=r2s.SIGMA_T,
                        beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=r2s.PHASE0_EPOCHS, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band", weight_mode="adaptive"))
        model = HardBCPINN(cfg).to(DEV)
        pde = ConvDiffPDE1D(cfg)
        cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = 1.0; cfg2["weight_mode"] = "adaptive"
        crit = RationalWeightingLoss(cfg2, DEV)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
        vis = PINNVisualizer(save_dir=OUT)
        tr = LayerTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td, layer_side="right")
        tr.train(cfg2, td, r2s.PHASE0_EPOCHS, 0, adaptive_freq=500)
        if not crit.field_frozen:
            crit.freeze_residual_field(model, pde)
        l2 = quick_l2(model, xt, un)
        cands.append(dict(seed=seed, l2=l2, gate=tr.gate_epoch,
                          state=copy.deepcopy(model.state_dict()),
                          t_field=crit._frozen_t_field.detach().clone(), sigma_end=model.get_sigma()))
        print(f"  [Phase0 pos{eps:g}] start {seed}: L2={l2:.3e} gate@{tr.gate_epoch}")
        tr.log_file.close()
    best = min(cands, key=lambda z: z["l2"])
    print(f"[Phase0 pos] 选用 start {best['seed']}, L2={best['l2']:.3e}")
    torch.save(best, path)
    return (best, cands) if return_all else best


# --------------------- pos：逐行复刻 mc.train_phase1_family ---------------------
def phase1_pos(eps, basin, family, beta, base_seed, xt, un, p1):
    r2s.set_seed(int(base_seed))
    cfg = pos_base_cfg(eps, f"p1_pos{eps:g}_s{base_seed}_{family}b{beta:g}")
    cfg.update(dict(lr=r2s.LR_PHASE1, use_gate=False, adam_epochs=p1, lbfgs_epochs=0,
                    use_best_ckpt=True, best_metric="max_L2_band", beta_schedule="const",
                    loss_beta=float(beta), beta_init=1.0, beta_start_step=0, beta_end_step=0,
                    fourier_scale=float(basin["sigma_end"]),
                    sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    model = HardBCPINN(cfg).to(DEV)
    pde = ConvDiffPDE1D(cfg)
    cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = float(beta); cfg2["weight_mode"] = "adaptive"
    crit = FamilyWeightingLoss(cfg2, DEV, family=family)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
    vis = PINNVisualizer(save_dir=OUT)
    td = {"x": xt, "u_true": un.flatten()}
    tr = LayerTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td, layer_side="right")
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=float(beta))
    tr.train(cfg2, td, p1, 0, adaptive_freq=500)
    ev = final_eval_pos(model, xt, un, eps)
    best_ep = int(getattr(tr, "best_epoch", -1)); tr.log_file.close()
    return ev, int(basin.get("gate", -1)), best_ep


# --------------------- bur：直接走权威 mc 路径（R2 金标准）---------------------
def run_bur(nu, seed, family, beta, k, p0, p1):
    xt, ut = r2s.build_test(nu)
    basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=15.0, k=k, phase0_epochs=p0)
    m, _dyn, _model, _crit, tr = mc.train_phase1_family(
        basin, nu, float(beta), seed, xt, ut, family=family, epochs=p1, tag=f"S2c_bur{nu:g}")
    tr.log_file.close()
    ev = dict(L2=float(m.get("L2_Error", np.nan)), Linf=float(m.get("L_inf_Error", np.nan)),
              L2_layer=float(m.get("l2_band", np.nan)), L2_outer=np.nan)
    return ev, float(basin["l2"]), int(basin.get("gate", -1)), int(getattr(tr, "best_epoch", -1))


def parse_cases(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            k, d = tok.split(":"); out.append((k, float(d)))
    return out


def parse_families(s):
    return [(lambda t: (t.split(":")[0], float(t.split(":")[1])))(t) for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--families", default="rational:1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p0", type=int, default=2000)
    ap.add_argument("--p1", type=int, default=8000)
    a = ap.parse_args()

    if a.baseline:
        cases = [("bur", 0.005), ("pos", 0.01), ("bur", 0.001), ("pos", 0.002)]
        seeds, fams = [0], [("rational", 1.0)]
        k, p0, p1 = (1, 60, 120) if mc.SMOKE else (a.k, a.p0, a.p1)
    else:
        cases = parse_cases(a.cases) or [("bur", 0.005), ("pos", 0.01)]
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        fams = parse_families(a.families)
        k, p0, p1 = (1, 60, 120) if mc.SMOKE else (a.k, a.p0, a.p1)

    csvp = os.path.join(OUT, "s2c_runs.csv")
    done = set()
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp, encoding="utf-8-sig")):
            done.add((r["kind"], float(r["delta"]), int(r["seed"]), r["family"], float(r["beta"])))
    newfile = not os.path.exists(csvp)
    fout = open(csvp, "a", newline="", encoding="utf-8-sig")
    wcsv = csv.DictWriter(fout, fieldnames=FIELDS)
    if newfile: wcsv.writeheader()
    n_new = 0
    try:
        for kind, delta in cases:
            xt, un = test_tensors(kind, delta)
            equiv = delta if kind == "bur" else delta / 2.0
            for seed in seeds:
                if kind == "bur":
                    basin_p0 = None
                else:
                    basin_p0 = enter_basin_pos(delta, seed, xt, un, k, p0)
                for family, beta in fams:
                    key = (kind, float(delta), int(seed), family, float(beta))
                    if key in done:
                        print("skip", key); continue
                    if kind == "bur":
                        ev, p0l2, gate, best_ep = run_bur(delta, seed, family, beta, k, p0, p1)
                    else:
                        ev, gate, best_ep = phase1_pos(delta, basin_p0, family, beta, seed, xt, un, p1)
                        p0l2 = float(basin_p0["l2"])
                    row = dict(kind=kind, delta=float(delta), equiv_nu=float(equiv), seed=seed,
                               family=family, beta=float(beta), p0=p0, p1=p1,
                               phase0_L2=p0l2, **ev, gate_epoch=gate, best_epoch=best_ep, k=k)
                    wcsv.writerow(row); fout.flush(); done.add(key); n_new += 1
                    print(f"[S2c] {kind} δ={delta:g}(ν_eq={equiv:g}) s{seed} {family}β{beta:g}: "
                          f"P0L2={p0l2:.2e} -> L2={ev['L2']:.2e} 内层={ev['L2_layer']:.2e} gate@{gate}")
    finally:
        fout.close()
    print(f"\n[S2c] 新训 {n_new} 组，结果 {csvp}")
    print("\n===== 同厚度配对（β=1，全域 L2 / 内层 L2）=====")
    allrows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    for nu_eq in sorted({float(r["equiv_nu"]) for r in allrows}):
        for kind in ["bur", "pos"]:
            sub = [r for r in allrows if abs(float(r["equiv_nu"]) - nu_eq) < 1e-12 and r["kind"] == kind]
            if not sub: continue
            L2 = np.median([float(r["L2"]) for r in sub]); Ll = np.median([float(r["L2_layer"]) for r in sub])
            p0l = np.median([float(r["phase0_L2"]) for r in sub])
            print(f"  ν_eq={nu_eq:<6} {kind}: Phase0={p0l:.2e}  全域L2={L2:.3e}  内层L2={Ll:.3e}  (n={len(sub)})")


if __name__ == "__main__":
    main()
