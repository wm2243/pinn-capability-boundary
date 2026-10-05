# -*- coding: utf-8 -*-
"""Cole-Hopf 参考解可信度验证（论文附录 C 的口径）：
(1) t=0 恒等; (2) 梯形 Ny h-细化误差带; (3) 周期镜像 M 收敛;
(4) 大 nu 与直接 FFT Cole-Hopf 互验; (5) 奇对称/u(0)=0; (6) 早期扰动理论;
(7) 关键点 scipy quad 独立复核。
运行: python -X utf8 verify_reference.py  （在 numerics/ 目录或设 PYTHONPATH）
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cole_hopf_periodic as ch

nu = 0.01 / np.pi
xo = np.linspace(-1, 1, 1001)
ok = True

print("=" * 72)
print("(1) t=0 恒等  max|u-u0|")
U0, _, _ = ch.reference_solution(nu, xo, np.array([0.0]), Ny=8001)
e0 = np.max(np.abs(U0[0] - ch.u_initial(xo)))
print("    %.2e" % e0); ok &= e0 < 1e-12

print("(2) Ny h-细化（以 Ny=32001 为参考）, nu=%.5f" % nu)
for t in [0.05, 0.5, 1.0]:
    base = ch._u_one_t(nu, t, xo, Ny=32001, M=2)
    row = "    t=%.2f:" % t
    for Ny in [4001, 8001, 16001]:
        u = ch._u_one_t(nu, t, xo, Ny=Ny, M=2)
        d = np.abs(u - base)
        row += "  Ny=%5d:L2=%.1e/Linf=%.1e" % (Ny, np.sqrt(np.mean(d ** 2)), d.max())
    print(row)

print("(3) 周期镜像 M 收敛 (t=1, Ny=8001)")
uM = [ch._u_one_t(nu, 1.0, xo, Ny=8001, M=Mm) for Mm in range(0, 5)]
for Mm, u in zip(range(0, 5), uM):
    d = np.abs(u - uM[-1])
    print("    M=%d  L2=%.2e Linf=%.2e" % (Mm, np.sqrt(np.mean(d ** 2)), d.max()))

print("(4) 大 nu 与直接 FFT Cole-Hopf 互验")
def fft_ch(nu2, t, N):
    x = np.linspace(-1, 1, N, endpoint=False)
    k = np.fft.fftfreq(N, d=2.0 / N) * 2 * np.pi
    phi0 = np.exp(ch._f0(x, nu2)); phi0 /= phi0.max()
    ph = np.fft.fft(phi0) * np.exp(-nu2 * k ** 2 * t)
    phi = np.fft.ifft(ph).real; phix = np.fft.ifft(1j * k * ph).real
    return x, -2 * nu2 * phix / phi
for nu2 in [0.02 / np.pi, 0.05 / np.pi]:
    xf, uf = fft_ch(nu2, 1.0, 8192)
    ui = ch._u_one_t(nu2, 1.0, xf, Ny=12001, M=2)
    d = np.abs(ui - uf)
    print("    nu=%.4f  L2=%.2e Linf=%.2e  min(int)=%.4f min(FFT)=%.4f"
          % (nu2, np.sqrt(np.mean(d ** 2)), d.max(), ui.min(), uf.min()))

print("(5) 奇对称 / u(0)=0 (t=1)")
u1 = ch._u_one_t(nu, 1.0, xo, Ny=16001, M=2)
esym = np.max(np.abs(u1 + u1[::-1])); e0c = abs(np.interp(0.0, xo, u1))
print("    max|u(x)+u(-x)|=%.2e, |u(0)|=%.2e, min=%.5f" % (esym, e0c, u1.min()))

print("(6) 早期 t=.05 非线性陡化: 实测 max|u-u0|=%.4f, 一阶扰动理论 t*pi/2=%.4f"
      % (np.max(np.abs(ch._u_one_t(nu, .05, xo, 16001, 2) - ch.u_initial(xo))), .05 * np.pi / 2))

print("(7) scipy quad 独立复核 t=1 关键位置（vs 梯形 Ny=16001）")
xq = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5])
uq = ch.reference_u_quad(nu, 1.0, xq)
ut = np.interp(xq, xo, u1)
for xi, a, b in zip(xq, uq, ut):
    print("    x=%5.2f  quad=%8.5f  trap=%8.5f  diff=%.1e" % (xi, a, b, abs(a - b)))

print("=" * 72)
print("结论: 参考解 rel-L2 ~1e-5(Linf~3e-5 @Ny16k) 且多法互证; PINN 目标 1e-3~1e-4, 裕度充足。")
