# -*- coding: utf-8 -*-
"""
S8: 时变零初始 Burgers — 减权 vs 梯度裁剪对照实验
验证：减权的本质是梯度尺度稳定器，和梯度裁剪作用类似，都不能改变盆地归属。

设置：
- 方程：u_t + u u_x = nu u_xx, nu=0.03/pi
- IC: u(x,0)=0, BC: u(-1,t)=-1, u(1,t)=1
- 优化器：纯 Adam 20000 步（不跑 L-BFGS，看 pas0 盆地归属）
- 每组 11 种子

组别：
  control:   beta=1,   无裁剪
  down_3:    beta=1/3, 无裁剪（中心区域减权）
  down_8:    beta=1/8, 无裁剪（更强减权）
  clip_0.1:  beta=1,   clipnorm=0.1
  clip_0.01: beta=1,   clipnorm=0.01

权重实现：基于位置的加权
  w(x) = 1 + (beta-1) * (1 - |x|)
  中心(x=0)权重=beta，边界(x=±1)权重=1
  beta<1 为减权，beta>1 为加权
"""
import os, sys, csv, time
from pathlib import Path
import numpy as np
import deepxde as dde

dde.config.set_default_float("float32")

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

NU = 0.03 / np.pi

# backend 检测
_BACKEND = dde.backend.backend_name
if _BACKEND == "pytorch":
    import torch
elif _BACKEND == "tensorflow":
    import tensorflow as tf


def gen_testdata():
    from scipy.io import loadmat
    data = loadmat(str(HERE / "true_solution" / "burgers_zeroIC_nu003pi.mat"))
    x = data["x"].flatten()
    t = data["t"].flatten()
    usol = data["usol"]
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = usol.T.flatten()[:, None]
    return X, y


def make_pde(beta):
    """生成加权 PDE 函数。beta<1 减权，beta>1 加权，beta=1 均匀。"""
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        res = dy_t + y * dy_x - NU * dy_xx
        # 位置加权：中心权重=beta，边界权重=1
        if _BACKEND == "pytorch":
            abs_x = torch.abs(x[:, 0:1])
        else:
            abs_x = tf.abs(x[:, 0:1])
        w = 1.0 + (beta - 1.0) * (1.0 - abs_x)
        return w * res
    return pde


def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], -1)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


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
        # 写表头
        with open(self.log_path, "w", newline="", encoding="utf-8-sig") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["step", "train_loss", "test_loss", "l2_relative"])

    def on_batch_end(self):
        self._step += 1
        if self._step % self.log_every == 0:
            try:
                train_loss = float(self.model.train_state.loss_train)
                test_loss = float(self.model.train_state.loss_test)
                l2 = float(self.model.train_state.metrics_test[0]) if self.model.train_state.metrics_test else 0.0
                self.rows.append([self._step, train_loss, test_loss, l2])
            except Exception:
                pass

    def on_train_end(self):
        import csv as _csv
        with open(self.log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f)
            w.writerows(self.rows)


def run_one(seed, beta=1.0, clipnorm=None, lr=1e-3, adam_steps=20000, save_dir=None):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_left = dde.icbc.DirichletBC(geomtime, lambda x: -1.0, boundary_left)
    bc_right = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_right)
    ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)

    X_test, y_test = gen_testdata()
    from scipy.spatial import cKDTree
    tree = cKDTree(X_test)
    def solution(X):
        _, idx = tree.query(X)
        return y_test[idx]

    data = dde.data.TimePDE(
        geomtime, make_pde(beta), [bc_left, bc_right, ic],
        num_domain=4000, num_boundary=100, num_initial=200,
        solution=solution, num_test=10000,
    )

    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # 编译
    if clipnorm is not None and _BACKEND == "tensorflow":
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        model.compile(opt, metrics=["l2 relative error"])
    else:
        model.compile("adam", lr=lr, metrics=["l2 relative error"])

    # PyTorch 梯度裁剪：包装 optimizer.step
    if clipnorm is not None and _BACKEND == "pytorch":
        original_step = model.opt.step
        def clipped_step(closure=None):
            torch.nn.utils.clip_grad_norm_(net.parameters(), clipnorm)
            return original_step(closure)
        model.opt.step = clipped_step

    # 训练过程记录
    callbacks = []
    if save_dir is not None:
        seed_dir = save_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_cb = TrainingLogCallback(seed_dir / "training_log.csv", log_every=500)
        callbacks.append(log_cb)

    model.train(iterations=adam_steps, display_every=5000, callbacks=callbacks if callbacks else None)

    y_pred = model.predict(X_test)
    l2 = np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test)

    # 盆地分类
    if l2 < 0.1:
        basin = "good"
    elif l2 < 2.0:
        basin = "intermediate"
    else:
        basin = "plateau"

    # 保存预测解
    if save_dir is not None:
        from scipy.io import savemat
        # 重建网格
        x_unique = np.unique(X_test[:, 0])
        t_unique = np.unique(X_test[:, 1])
        u_grid = y_pred.reshape(len(t_unique), len(x_unique)).T  # (Nx, Nt)
        u_true_grid = y_test.reshape(len(t_unique), len(x_unique)).T
        savemat(str(seed_dir / "prediction.mat"), {
            "x": x_unique,
            "t": t_unique,
            "u_pred": u_grid,
            "u_true": u_true_grid,
            "L2_relative": l2,
            "basin": basin,
            "beta": beta,
            "clipnorm": clipnorm if clipnorm else 0,
            "seed": seed,
        })

    print(f"[seed={seed}] beta={beta:.4f}, clip={clipnorm}, L2={l2:.4e}, basin={basin}", flush=True)
    return l2, basin


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=str, default="control",
                    choices=["control", "down_3", "down_8", "clip_0.1", "clip_0.01"])
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--adam_steps", type=int, default=20000)
    args = ap.parse_args()

    configs = {
        "control":   dict(beta=1.0, clipnorm=None, lr=1e-3),
        "down_3":    dict(beta=1/3, clipnorm=None, lr=1e-3),
        "down_8":    dict(beta=1/8, clipnorm=None, lr=1e-3),
        "clip_0.1":  dict(beta=1.0, clipnorm=0.1, lr=1e-3),
        "clip_0.01": dict(beta=1.0, clipnorm=0.01, lr=1e-3),
    }
    cfg = configs[args.group]

    # 保存目录：每组一个文件夹，内含每个种子的训练日志和预测解
    save_dir = OUT / f"s8_zeroIC_{args.group}_K{args.K}_adam{args.adam_steps}"
    save_dir.mkdir(parents=True, exist_ok=True)

    fn = save_dir / "summary.csv"
    fieldnames = ["seed", "group", "beta", "clipnorm", "L2_relative", "basin"]
    new_file = not fn.exists()

    with open(fn, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for k in range(args.K):
            seed = args.base_seed * 100 + k
            t0 = time.time()
            l2, basin = run_one(seed, adam_steps=args.adam_steps, save_dir=save_dir, **cfg)
            w.writerow(dict(seed=seed, group=args.group, beta=cfg["beta"],
                            clipnorm=cfg["clipnorm"], L2_relative=l2, basin=basin))
            f.flush()
            print(f"[k={k}] seed={seed} L2={l2:.4e} basin={basin} ({time.time()-t0:.1f}s)", flush=True)

    # 统计
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    l2s = [float(r["L2_relative"]) for r in rows]
    basins = [r["basin"] for r in rows]
    print(f"\n=== {args.group} (K={len(rows)}) ===")
    print(f"L2 range: [{min(l2s):.3e}, {max(l2s):.3e}], median={np.median(l2s):.3e}")
    for b in ["good", "intermediate", "plateau"]:
        cnt = basins.count(b)
        print(f"  {b}: {cnt}/{len(rows)} ({100*cnt/len(rows):.0f}%)")


if __name__ == "__main__":
    main()
