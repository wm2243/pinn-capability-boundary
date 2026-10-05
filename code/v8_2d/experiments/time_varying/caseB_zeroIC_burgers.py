# -*- coding: utf-8 -*-
"""
算例 B：零初始时变 Burgers（平台区 / 多峰结构）
================================================
方程：u_t + u u_x = ν u_xx,  ν = 0.03/π,  x∈[-1,1], t∈[0,1]
IC: u(x,0) = 0,  BC: u(-1,t) = -1, u(1,t) = 1  （零初始，存在平台区）
真解：burgers_zeroIC_nu003pi.mat（Crank-Nicolson + Picard 隐式有限差分，Nx=1024, dt=1e-4）
优化器：纯 Adam（准入算法范围内，无 L-BFGS）

包含两个实验，直接运行本脚本依次完成全部：
  B1 空间加权 / 减权（能否跳出平台区）
      - 分组：control(β=1), down_3(β=1/3), down_8(β=1/8), up_3(β=3), up_8(β=8)
      - 权重：w(x) = 1 + (β-1)(1-|x|)，中心权重=β，边界权重=1
      - 每组 11 种子（seed 200~210，base_seed=2，与 S7 对齐），纯 Adam 20000 步
  B2 梯度裁剪（绝对范数限制，能否跳出平台区）
      - 分组：control(无裁剪), clip_0.1(max_norm=0.1), clip_0.01(max_norm=0.01)
      - 用 ClipAdam（继承 torch.optim.Adam，step 前 clip_grad_norm_）
      - 每组 11 种子（seed 200~210），纯 Adam 20000 步
      - 训练日志额外记录梯度范数

与 B1 的区别：
  - B1 改变各点梯度的相对尺度（空间加权，R 层干预）
  - B2 限制梯度的绝对范数（全局稳定，θ 层干预）
两者共同验证：准入的固定步长一阶算法内，相对尺度或绝对范数干预都不能可靠跳出平台区。

用法：
    python caseB_zeroIC_burgers.py                  # 全量（B1 5组 + B2 3组，各11种子）
    python caseB_zeroIC_burgers.py --skip_b1        # 只跑 B2
    python caseB_zeroIC_burgers.py --b1_groups control down_3  # 只跑指定组
结果保存到 results/caseB/ 下。
"""
import csv, time, argparse
from pathlib import Path
import numpy as np
import deepxde as dde
import torch

dde.config.set_default_float("float32")

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "caseB"
OUT.mkdir(parents=True, exist_ok=True)

# ===================== 全局配置 =====================
NU = 0.03 / np.pi
NET_LAYERS = [2] + [50] * 4 + [1]
LR = 1e-3
GOOD_BASIN_THRESH = 0.1


# ===================== ClipAdam =====================
class ClipAdam(torch.optim.Adam):
    """Adam with gradient clipping before each step（B2 用）。"""
    def __init__(self, params, max_norm=None, **kwargs):
        super().__init__(params, **kwargs)
        self.max_norm = max_norm

    def step(self, closure=None):
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.param_groups[0]["params"] if p.grad is not None],
                max_norm=self.max_norm,
            )
        return super().step(closure)


# ===================== 公共函数 =====================
def gen_true_solution():
    """加载零初始真解 burgers_zeroIC_nu003pi.mat。"""
    from scipy.io import loadmat
    data = loadmat(str(HERE / "true_solution" / "burgers_zeroIC_nu003pi.mat"))
    x = data["x"].flatten()
    t = data["t"].flatten()
    usol = data["usol"]
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = usol.T.flatten()[:, None]
    return X, y


def make_pde_weighted(beta):
    """加权 PDE（B1 用）。β=1 均匀，β>1 加权，β<1 减权。"""
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        res = dy_t + y * dy_x - NU * dy_xx
        w = 1.0 + (beta - 1.0) * (1.0 - torch.abs(x[:, 0:1]))
        return w * res
    return pde


def pde_plain(x, y):
    """无加权 PDE（B2 用）。"""
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - NU * dy_xx


class TrainingLogCallback(dde.callbacks.Callback):
    """记录训练过程；log_grad_norm=True 时额外记录梯度范数（B2 用）。"""
    def __init__(self, log_path, log_every=500, log_grad_norm=False):
        super().__init__()
        self.log_path = log_path
        self.log_every = log_every
        self.log_grad_norm = log_grad_norm
        self.rows = []
        self._step = 0

    def on_train_begin(self):
        self._step = 0
        header = ["step", "train_loss", "test_loss", "l2_relative"]
        if self.log_grad_norm:
            header.append("grad_norm")
        with open(self.log_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(header)

    def on_batch_end(self):
        self._step += 1
        if self._step % self.log_every == 0:
            try:
                st = self.model.train_state
                row = [self._step, float(st.loss_train), float(st.loss_test),
                       float(st.metrics_test[0]) if st.metrics_test else 0.0]
                if self.log_grad_norm:
                    gn = 0.0
                    for p in self.model.net.parameters():
                        if p.grad is not None:
                            gn += p.grad.data.norm(2).item() ** 2
                    row.append(gn ** 0.5)
                self.rows.append(row)
            except Exception:
                pass

    def on_train_end(self):
        with open(self.log_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(self.rows)


def build_model_zeroIC(seed, pde_func=None):
    """构建零初始 Burgers 的数据、网络、模型（不训练，不绑定优化器）。
    pde_func: PDE 残差函数，默认 pde_plain（无加权）；B1 传入 make_pde_weighted(β)。"""
    if pde_func is None:
        pde_func = pde_plain
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def boundary_left(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)
    def boundary_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    bc_left = dde.icbc.DirichletBC(geomtime, lambda x: -1, boundary_left)
    bc_right = dde.icbc.DirichletBC(geomtime, lambda x: 1, boundary_right)
    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    X_test, y_test = gen_true_solution()
    from scipy.spatial import cKDTree
    tree = cKDTree(X_test)
    def solution(X):
        _, idx = tree.query(X)
        return y_test[idx]

    data = dde.data.TimePDE(
        geomtime, pde_func, [bc_left, bc_right, ic],
        num_domain=4000, num_boundary=100, num_initial=200,
        solution=solution, num_test=10000,
    )
    net = dde.nn.FNN(NET_LAYERS, "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    return model, data, X_test, y_test


def save_prediction(save_dir, seed, X_test, y_test, y_pred, **extra):
    from scipy.io import savemat
    seed_dir = save_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    x_unique = np.unique(X_test[:, 0])
    t_unique = np.unique(X_test[:, 1])
    u_grid = y_pred.reshape(len(t_unique), len(x_unique)).T
    u_true_grid = y_test.reshape(len(t_unique), len(x_unique)).T
    payload = {"x": x_unique, "t": t_unique,
               "u_pred": u_grid, "u_true": u_true_grid, "seed": seed}
    payload.update(extra)
    savemat(str(seed_dir / "prediction.mat"), payload)


def done_seeds_of(fn, value_col="L2_relative"):
    done = set()
    if fn.exists() and fn.stat().st_size > 0:
        for r in csv.DictReader(open(fn, encoding="utf-8-sig")):
            try:
                if np.isfinite(float(r[value_col])):
                    done.add(int(r["seed"]))
            except Exception:
                pass
    return done


def write_header_if_needed(fn, fieldnames):
    if (not fn.exists()) or (fn.stat().st_size == 0):
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(fieldnames)


# ===================== B1：空间加权 / 减权 =====================
B1_BETA_MAP = {
    "control": 1.0,
    "down_3": 1.0 / 3,
    "down_8": 1.0 / 8,
    "up_3": 3.0,
    "up_8": 8.0,
}


def run_b1_one(seed, beta, adam_steps=20000, save_dir=None):
    model, data, X_test, y_test = build_model_zeroIC(seed, pde_func=make_pde_weighted(beta))
    callbacks = []
    if save_dir is not None:
        seed_dir = save_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(TrainingLogCallback(seed_dir / "training_log.csv"))
    model.compile("adam", lr=LR, metrics=["l2 relative error"])
    model.train(iterations=adam_steps, display_every=5000,
                callbacks=callbacks if callbacks else None)
    y_pred = model.predict(X_test)
    l2 = np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test)
    if save_dir is not None:
        save_prediction(save_dir, seed, X_test, y_test, y_pred,
                        L2_relative=l2, beta=beta)
    print(f"  [seed={seed}] β={beta:.3f}, L2={l2:.4e}", flush=True)
    return l2


def run_b1(groups=("control", "down_3", "down_8", "up_3", "up_8"),
           K=11, base_seed=2, adam_steps=20000):
    summary = []
    for group in groups:
        beta = B1_BETA_MAP[group]
        save_dir = OUT / f"b1_{group}_K{K}_adam{adam_steps}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fn = save_dir / "summary.csv"
        fields = ["seed", "group", "beta", "L2_relative"]
        write_header_if_needed(fn, fields)
        done = done_seeds_of(fn)
        with open(fn, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            for k in range(K):
                seed = base_seed * 100 + k
                if seed in done:
                    print(f"[B1 {group}] seed={seed} 已完成，跳过", flush=True)
                    continue
                t0 = time.time()
                l2 = run_b1_one(seed, beta, adam_steps=adam_steps, save_dir=save_dir)
                w.writerow({"seed": seed, "group": group, "beta": beta, "L2_relative": l2})
                f.flush()
                print(f"[B1 {group}] k={k} seed={seed} L2={l2:.4e} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
        l2s = [float(r["L2_relative"]) for r in rows]
        good = sum(1 for l in l2s if l < GOOD_BASIN_THRESH)
        summary.append((group, beta, len(l2s), np.median(l2s), np.mean(l2s),
                        np.min(l2s), np.max(l2s), good))
        print(f"\n[B1 {group}] β={beta:.3f}, K={len(l2s)}, "
              f"median={np.median(l2s):.3e}, 好盆地={good}/{len(l2s)}\n", flush=True)
    print(f"\n{'='*70}\nB1 全组对比（好盆地阈值 L2<{GOOD_BASIN_THRESH}）\n{'='*70}")
    print(f"{'group':>10} {'β':>7} {'K':>4} {'median':>10} {'mean':>10} "
          f"{'min':>10} {'max':>10} {'好盆地':>8}")
    for group, beta, k, med, mean, lo, hi, good in summary:
        print(f"{group:>10} {beta:7.3f} {k:4d} {med:10.3e} {mean:10.3e} "
              f"{lo:10.3e} {hi:10.3e} {good:3d}/{k:<3d}")


# ===================== B2：梯度裁剪 =====================
B2_CLIP_MAP = {"control": None, "clip_0.1": 0.1, "clip_0.01": 0.01}


def run_b2_one(seed, clip_norm, adam_steps=20000, save_dir=None):
    model, data, X_test, y_test = build_model_zeroIC(seed)
    callbacks = []
    if save_dir is not None:
        seed_dir = save_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(TrainingLogCallback(
            seed_dir / "training_log.csv", log_grad_norm=True))
    model.compile("adam", lr=LR, metrics=["l2 relative error"])
    if clip_norm is not None:
        model.opt = ClipAdam(model.net.parameters(), max_norm=clip_norm, lr=LR)
    model.train(iterations=adam_steps, display_every=5000,
                callbacks=callbacks if callbacks else None)
    y_pred = model.predict(X_test)
    l2 = np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test)
    if save_dir is not None:
        save_prediction(save_dir, seed, X_test, y_test, y_pred,
                        L2_relative=l2, clip_norm=clip_norm if clip_norm else 0.0)
    label = f"clip_{clip_norm}" if clip_norm else "control"
    print(f"  [seed={seed}] {label}, L2={l2:.4e}", flush=True)
    return l2


def run_b2(groups=("control", "clip_0.1", "clip_0.01"),
           K=11, base_seed=2, adam_steps=20000):
    summary = []
    for group in groups:
        clip_norm = B2_CLIP_MAP[group]
        save_dir = OUT / f"b2_{group}_K{K}_adam{adam_steps}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fn = save_dir / "summary.csv"
        fields = ["seed", "group", "clip_norm", "L2_relative"]
        write_header_if_needed(fn, fields)
        done = done_seeds_of(fn)
        with open(fn, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            for k in range(K):
                seed = base_seed * 100 + k
                if seed in done:
                    print(f"[B2 {group}] seed={seed} 已完成，跳过", flush=True)
                    continue
                t0 = time.time()
                l2 = run_b2_one(seed, clip_norm, adam_steps=adam_steps, save_dir=save_dir)
                w.writerow({"seed": seed, "group": group,
                            "clip_norm": clip_norm if clip_norm else 0.0,
                            "L2_relative": l2})
                f.flush()
                print(f"[B2 {group}] k={k} seed={seed} L2={l2:.4e} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
        l2s = [float(r["L2_relative"]) for r in rows]
        good = sum(1 for l in l2s if l < GOOD_BASIN_THRESH)
        summary.append((group, clip_norm, len(l2s), np.median(l2s), np.mean(l2s),
                        np.min(l2s), np.max(l2s), good))
        print(f"\n[B2 {group}] clip={clip_norm}, K={len(l2s)}, "
              f"median={np.median(l2s):.3e}, 好盆地={good}/{len(l2s)}\n", flush=True)
    print(f"\n{'='*70}\nB2 全组对比（好盆地阈值 L2<{GOOD_BASIN_THRESH}）\n{'='*70}")
    print(f"{'group':>10} {'clip':>7} {'K':>4} {'median':>10} {'mean':>10} "
          f"{'min':>10} {'max':>10} {'好盆地':>8}")
    for group, cn, k, med, mean, lo, hi, good in summary:
        print(f"{group:>10} {str(cn):>7} {k:4d} {med:10.3e} {mean:10.3e} "
              f"{lo:10.3e} {hi:10.3e} {good:3d}/{k:<3d}")


# ===================== main =====================
def main():
    ap = argparse.ArgumentParser(description="算例B：零初始时变 Burgers（B1加权减权 + B2梯度裁剪）")
    ap.add_argument("--skip_b1", action="store_true", help="跳过 B1")
    ap.add_argument("--skip_b2", action="store_true", help="跳过 B2")
    ap.add_argument("--b1_groups", type=str, nargs="+",
                    default=["control", "down_3", "down_8", "up_3", "up_8"],
                    choices=list(B1_BETA_MAP.keys()))
    ap.add_argument("--b2_groups", type=str, nargs="+",
                    default=["control", "clip_0.1", "clip_0.01"],
                    choices=list(B2_CLIP_MAP.keys()))
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--base_seed", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=20000)
    args = ap.parse_args()

    print(f"结果目录：{OUT}")
    if not args.skip_b1:
        print("\n" + "#"*70)
        print("# B1：空间加权 / 减权（能否跳出平台区）")
        print("#"*70)
        run_b1(groups=tuple(args.b1_groups), K=args.K,
               base_seed=args.base_seed, adam_steps=args.adam_steps)
    if not args.skip_b2:
        print("\n" + "#"*70)
        print("# B2：梯度裁剪（能否跳出平台区）")
        print("#"*70)
        run_b2(groups=tuple(args.b2_groups), K=args.K,
               base_seed=args.base_seed, adam_steps=args.adam_steps)
    print("\n算例 B 全部完成。")


if __name__ == "__main__":
    main()
