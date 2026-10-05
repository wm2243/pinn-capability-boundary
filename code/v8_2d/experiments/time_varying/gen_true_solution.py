# -*- coding: utf-8 -*-
"""
时变Burgers真解生成（迎风格式 + 前向欧拉）
u_t + u u_x = nu u_xx, nu=0.01/pi
x in [-1,1], t in [0,0.99]
IC: u(x,0) = -sin(pi x), BC: u(±1,t)=0
"""
import numpy as np
from pathlib import Path
import time

OUT = Path(__file__).parent / "true_solution"
OUT.mkdir(exist_ok=True)


def generate_true_solution(nu=0.01 / np.pi, Nx=4096, dt=2e-6, T=0.99):
    x_full = np.linspace(-1, 1, Nx + 2)
    dx = x_full[1] - x_full[0]
    x = x_full[1:-1]
    u = -np.sin(np.pi * x)

    cfl = dt * 1.0 / dx
    visc = dt * nu / dx**2
    print(f"dx={dx:.4e}, dt={dt:.1e}, CFL={cfl:.3f}, visc={visc:.3f}")

    n_steps = int(T / dt)
    save_every = max(1, n_steps // 100)
    n_save = n_steps // save_every + 1
    t_save = np.zeros(n_save)
    u_save = np.zeros((n_save, Nx + 2))
    u_save[0, 1:-1] = u

    t0 = time.time()
    save_idx = 1
    for n in range(n_steps):
        # 迎风格式
        u_left = np.concatenate([[0], u[:-1]])
        u_right = np.concatenate([u[1:], [0]])
        ux = np.where(u >= 0, (u - u_left) / dx, (u_right - u) / dx)
        uxx = (u_right - 2 * u + u_left) / dx**2
        u = u - dt * u * ux + dt * nu * uxx
        u[0] = 0; u[-1] = 0

        if (n + 1) % save_every == 0 and save_idx < n_save:
            t_save[save_idx] = (n + 1) * dt
            u_save[save_idx, 1:-1] = u
            save_idx += 1

    u_save = u_save[:save_idx]
    t_save = t_save[:save_idx]
    print(f"  done in {time.time()-t0:.1f}s, {n_steps} steps, {save_idx} snapshots")

    print(f"  t=0:    u range [{u_save[0].min():.4f}, {u_save[0].max():.4f}]")
    print(f"  t=0.5:  u range [{u_save[len(u_save)//2].min():.4f}, {u_save[len(u_save)//2].max():.4f}]")
    print(f"  t=0.99: u range [{u_save[-1].min():.4f}, {u_save[-1].max():.4f}]")

    np.savez(OUT / "burgers_true.npz", x=x_full, t=t_save, u=u_save)
    print(f"  saved to {OUT / 'burgers_true.npz'}")


if __name__ == "__main__":
    generate_true_solution()
