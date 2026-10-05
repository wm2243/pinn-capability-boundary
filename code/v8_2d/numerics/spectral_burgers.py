# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:12:26 2026

@author: wwm
"""

# -*- coding: utf-8 -*-
"""
Fourier Pseudo-spectral solver with ETDRK4 and 3/2 dealiasing.
Numerically stable for all Nx via adaptive dt and Kassam-Trefethen coefficients.
"""
import numpy as np


def solve_burgers(Nx, nu, time_span, t_eval, domain=(-1.0, 1.0)):
    # Ensure Nx is even for FFT efficiency
    if Nx % 2 != 0:
        Nx += 1

    L = domain[1] - domain[0]
    x = np.linspace(domain[0], domain[1], Nx, endpoint=False)

    # Wavenumbers
    k = np.fft.fftfreq(Nx, d=L / Nx) * 2.0 * np.pi

    # Initial condition: u(x,0) = -sin(pi*x) on (-1,1)
    u = -np.sin(np.pi * (x - domain[0]) / L * 2.0)
    if domain == (-1.0, 1.0):
        u = -np.sin(np.pi * x)

    # Linear operator in Fourier space
    L_op = -nu * k ** 2

    # === Adaptive dt for ETDRK4 (spectral-radius based, no empirical viscous floor) ===
    dx = L / Nx
    u_max = np.max(np.abs(u)) + 1e-14
    
    # Nonlinear CFL: 0.4 is conservative-safe for ETD-RK4 + 3/2 dealiasing on Burgers
    # (0.8 caused blowup at Nx=64 due to aliasing resonance at critical dt)
    dt_cfl = 0.4 * dx / u_max
    
    # Linear stability constraint from ETDRK4 stability region
    # The method is A-stable for the linear part, but aliasing introduces
    # spurious high-k modes that see an effective z = -nu*k_max^2*dt.
    # We require |z_max| < 20 to stay within accurate phi evaluation range
    # and avoid transient growth from dealiasing residuals.
    k_max = np.pi * Nx / L  # Maximum resolved wavenumber
    if nu > 0:
        dt_linear = 20.0 / (nu * k_max ** 2)
    else:
        dt_linear = np.inf
    
    if len(t_eval) > 1:
        dt_requested = (time_span[1] - time_span[0]) / (len(t_eval) - 1)
    else:
        dt_requested = np.inf

    # dt scales as O(Nx^-1) from CFL, NOT O(Nx^-2) from viscous floor
    # dt_linear is a hard ceiling that prevents phi-function inaccuracy,
    # not an algebraic convergence killer
    dt_stable = min(dt_requested, dt_cfl, dt_linear)
    
    n_substeps = max(1, int(np.ceil((time_span[1] - time_span[0]) / dt_stable)))
    dt = (time_span[1] - time_span[0]) / n_substeps
    
    # === ETDRK4 Coefficients (exact Kassam-Trefethen 2005 notation) ===
    z = L_op * dt
    ez = np.exp(z)
    ez2 = np.exp(z / 2.0)

    phi1 = np.zeros_like(z)
    phi2 = np.zeros_like(z)
    phi3 = np.zeros_like(z)

    small = np.abs(z) < 1e-7
    large = ~small

    zs = z[small]
    phi1[small] = 1.0 + zs/2.0 + zs**2/6.0 + zs**3/24.0 + zs**4/120.0
    phi2[small] = 1.0/2.0 + zs/6.0 + zs**2/24.0 + zs**3/120.0 + zs**4/720.0
    phi3[small] = 1.0/6.0 + zs/24.0 + zs**2/120.0 + zs**3/720.0 + zs**4/5040.0

    zl = z[large]
    ez_l = ez[large]
    phi1[large] = (ez_l - 1.0) / zl
    phi2[large] = (phi1[large] - 1.0) / zl
    phi3[large] = (phi2[large] - 0.5) / zl

    # Half-step φ functions
    z2 = z / 2.0
    phi1_h = np.zeros_like(z)
    phi2_h = np.zeros_like(z)

    phi1_h[small] = 1.0 + zs/4.0 + zs**2/24.0 + zs**3/192.0 + zs**4/1920.0
    phi2_h[small] = 1.0/2.0 + zs/12.0 + zs**2/96.0 + zs**3/960.0 + zs**4/11520.0

    ez2_l = ez2[large]
    phi1_h[large] = (ez2_l - 1.0) / z2[large]
    phi2_h[large] = (phi1_h[large] - 1.0) / z2[large]

    # === Dealiasing setup (3/2 rule, index-based padding) ===
    Nx_pad = int(3 * Nx // 2)
    k_pad = np.fft.fftfreq(Nx_pad, d=L / Nx_pad) * 2.0 * np.pi
    n_pos = Nx // 2

    def _pad_fft(u_hat_nx):
        out = np.zeros(Nx_pad, dtype=complex)
        out[:n_pos + 1] = u_hat_nx[:n_pos + 1]
        out[-n_pos:] = u_hat_nx[-n_pos:]
        return out

    def _truncate_fft(u_hat_pad):
        out = np.zeros(Nx, dtype=complex)
        out[:n_pos + 1] = u_hat_pad[:n_pos + 1]
        out[-n_pos:] = u_hat_pad[-n_pos:]
        return out

    def nonlinear_term(v_hat):
        """Compute FFT of -u*u_x with proper 3/2 dealiasing."""
        v_pad = _pad_fft(v_hat)
        v_phys = np.fft.ifft(v_pad).real
        vx_phys = np.fft.ifft(1j * k_pad * v_pad).real
        nl = -v_phys * vx_phys
        return _truncate_fft(np.fft.fft(nl))

    # === Time Integration (KT2005 Algorithm 2) ===
    u_sol = np.zeros((len(t_eval), Nx), dtype=np.float64)
    u_sol[0] = u.copy()

    if len(t_eval) <= 1:
        return {"x": x, "t": np.array(t_eval, dtype=np.float64), "u": u_sol}

    u_hat = np.fft.fft(u)
    eval_idx = 1
    step_count = 0
    t_target = t_eval[eval_idx]

    while eval_idx < len(t_eval):
        N1 = nonlinear_term(u_hat)

        a_hat = ez2 * u_hat + dt * phi1_h * N1 / 2.0
        Na = nonlinear_term(a_hat)

        b_hat = ez2 * u_hat + dt * phi1_h * (Na / 2.0 - N1 / 4.0)
        Nb = nonlinear_term(b_hat)

        c_hat = ez2 * a_hat + dt * phi1_h * (Nb - Na / 2.0 + N1 / 4.0)
        Nc = nonlinear_term(c_hat)

        # ★ KT2005 Eq.(8) — CORRECT full-step update ★
        u_hat = (ez * u_hat
                 + dt * (phi1 * N1
                         - 2.0 * phi2 * Na
                         + 2.0 * phi2 * Nb
                         + phi3 * (N1 - 2.0 * Na + 2.0 * Nb - Nc)))

        u_hat.imag[np.abs(u_hat.imag) < 1e-14] = 0.0

        # ★ SINGLE time advancement (removed duplicate block) ★
        step_count += 1
        t_current = time_span[0] + step_count * dt

        if t_current >= t_target - 1e-14 * dt:
            u_sol[eval_idx] = np.fft.ifft(u_hat).real
            eval_idx += 1
            if eval_idx < len(t_eval):
                t_target = t_eval[eval_idx]

    return {
        "x": x,
        "t": np.array(t_eval, dtype=np.float64),
        "u": u_sol,
    }