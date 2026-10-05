# -*- coding: utf-8 -*-
"""
S9/A2: 时变 Burgers 加权 β 扫描（非零初始，好盆地内，纯 Adam）
验证：加权 β∈[3,15] 在时变好盆地内是否同样有效。

设置：
- 方程：u_t + u u_x = nu u_xx, nu=0.01/pi
- IC: u(x,0)=-sin(pi x), BC: u(±1,t)=0（标准设置，好盆地）
- 优化器：纯 Adam 50000 步（准入算法范围内，无 L-BFGS）
- β = {1, 3, 8, 15}，每组 11 种子
- 记录：最终 L2、训练过程、预测解

权重实现：基于位置的加权
  w(x) = 1 + (beta-1) * (1 - |x|)
  中心(x=0)权重=beta，边界(x=±1)权重=1
"""
import os, sys, csv, time
from pathlib import Path
import numpy as np
import deepxde as dde

dde.config.set_default_float("float32")

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

NU = 0.01 / np.pi

_BACKEND = dde.backend.backend_name
if _BACKEND == "pytorch":
    import torch
elif _BACKEND == "tensorflow":
    import tensorflow as tf


def gen_testdata():
    """加载 Raissi 标准真解。"""
    from scipy.io import loadmat
    data = loadmat(str(HERE / "true_solution" / "burgers_shock.mat"))
    x = data["x"].flatten()
    t = data["t"].flatten()
    usol = data["usol"]
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = usol.T.flatten()[:, None]
    return X, y


def make_pde(beta):
    """生成加权 PDE 函数。beta>1 加权，beta=1 均匀。"""
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        res = dy_t + y * dy_x - NU * dy_xx
        if _BACKEND == "pytorch":
            abs_x = torch.abs(x[:, 0:1])
        else:
            abs_x = tf.abs(x[:, 0:1])
        w = 1.0 + (beta - 1.0) * (1.0 - abs_x)
        return w * res
    return pde


class TrainingLogCallback(dde.callbacks.Callback):
    """记录训练过程指标到CSV。"""
    def __init__(self, log_path, log_every=500):
        super().__init__()
        self.log_path = log_path
        self.log_every = log_every
        self.rows = []
        self._step = 0

    def on_train_begin(self):
        self._step = 0
        import csv as _csv
        with open(self.log_path, "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f)
            w.writerow(["step", "phase", "train_loss", "test_loss", "l2_relative"])

    def on_batch_end(self):
        self._step += 1
        if self._step % self.log_every == 0:
            try:
                train_loss = float(self.model.train_state.loss_train)
                test_loss = float(self.model.train_state.loss_test)
                l2 = float(self.model.train_state.metrics_test[0]) if self.model.train_state.metrics_test else 0.0
                phase = "adam"
                self.rows.append([self._step, phase, train_loss, test_loss, l2])
            except Exception:
                pass

    def on_train_end(self):
        import csv as _csv
        with open(self.log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f)
            w.writerows(self.rows)


def run_one(seed, beta=15.0, adam_steps=50000, save_dir=None):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]),
                    lambda _, on_initial: on_initial)

    X_test, y_test = gen_testdata()
    from scipy.spatial import cKDTree
    tree = cKDTree(X_test)
    def solution(X):
        _, idx = tree.query(X)
        return y_test[idx]

    data = dde.data.TimePDE(
        geomtime, make_pde(beta), [bc, ic],
        num_domain=4000, num_boundary=100, num_initial=200,
        solution=solution, num_test=10000,
    )

    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # 训练过程记录
    callbacks = []
    if save_dir is not None:
        seed_dir = save_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_cb = TrainingLogCallback(seed_dir / "training_log.csv", log_every=500)
        callbacks.append(log_cb)

    # 纯 Adam（准入算法范围内，无 L-BFGS）
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(iterations=adam_steps, display_every=5000, callbacks=callbacks if callbacks else None)

    y_pred = model.predict(X_test)
    l2 = np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test)

    # 保存预测解
    if save_dir is not None:
        from scipy.io import savemat
        x_unique = np.unique(X_test[:, 0])
        t_unique = np.unique(X_test[:, 1])
        u_grid = y_pred.reshape(len(t_unique), len(x_unique)).T
        u_true_grid = y_test.reshape(len(t_unique), len(x_unique)).T
        savemat(str(seed_dir / "prediction.mat"), {
            "x": x_unique,
            "t": t_unique,
            "u_pred": u_grid,
            "u_true": u_true_grid,
            "L2_relative": l2,
            "beta": beta,
            "seed": seed,
        })

    print(f"[seed={seed}] beta={beta:.1f}, L2={l2:.4e}", flush=True)
    return l2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0, help="加权参数 beta")
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--adam_steps", type=int, default=50000)
    args = ap.parse_args()

    save_dir = OUT / f"a2_weighted_beta{args.beta}_K{args.K}_adam{args.adam_steps}"
    save_dir.mkdir(parents=True, exist_ok=True)

    fn = save_dir / "summary.csv"
    fieldnames = ["seed", "beta", "L2_relative"]

    # 断点续跑：读取已完成的 seed（且 L2 非空/非 nan），避免重复训练
    done_seeds = set()
    if fn.exists():
        for r in csv.DictReader(open(fn, encoding="utf-8-sig")):
            try:
                v = float(r["L2_relative"])
                if np.isfinite(v):
                    done_seeds.add(int(r["seed"]))
            except Exception:
                pass
    # 文件不存在或为空时都需要写表头（防止空文件残留导致缺表头）
    need_header = (not fn.exists()) or (fn.stat().st_size == 0)

    with open(fn, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            w.writeheader()
        for k in range(args.K):
            seed = args.base_seed * 100 + k
            if seed in done_seeds:
                print(f"[k={k}] seed={seed} 已完成，跳过", flush=True)
                continue
            t0 = time.time()
            l2 = run_one(seed, beta=args.beta, adam_steps=args.adam_steps,
                         save_dir=save_dir)
            w.writerow(dict(seed=seed, beta=args.beta, L2_relative=l2))
            f.flush()
            print(f"[k={k}] seed={seed} L2={l2:.4e} ({time.time()-t0:.1f}s)", flush=True)

    # 统计
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    l2s = [float(r["L2_relative"]) for r in rows]
    print(f"\n=== beta={args.beta} (K={len(rows)}) ===")
    print(f"L2 range: [{min(l2s):.3e}, {max(l2s):.3e}], median={np.median(l2s):.3e}, mean={np.mean(l2s):.3e}")


if __name__ == "__main__":
    main()
