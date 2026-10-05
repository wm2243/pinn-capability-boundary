# Experiment Registry: paper numbers <-> repo codenames <-> scripts

This table is the authoritative mapping between the experiment numbers used in the paper and the internal codenames used in this repository. The paper text and figures use "Experiment X.Y (descriptive name)"; repo scripts and data directories keep the codenames. Provenance rows refer to the `id` column of `claims_provenance.csv`.

Numbering rule: X = paper section number, Y = order of first appearance within that section.

## Main table: numbered experiments

| Paper number | Descriptive name | Repo codename | Generating script(s) | Aggregated data directory | Provenance rows |
|---|---|---|---|---|---|
| Experiment 4.1 | synthetic equicorrelation check | P8 | `exp_V2_P8_synthetic_equicorr.py` | `data/aggregated/exp_V2_P8_synthetic_equicorr/` | C23–C25 |
| Experiment 4.2 | frozen-Jacobian spectral scan | F1d | `exp_V2_F1d_real_jacobian.py`, `exp_V2_F1d_N1024_verify.py`, `exp_V2_F1d_largeN_fix.py`, `exp_V2_F1d_lmin_verify.py` | `data/aggregated/v7_v2_exp_V2_F1d_*/`, `data/aggregated/exp_V2_P1_constrained/` | C26–C31, C36, C38–C39 |
| Experiment 4.3 | in-training spectral dynamics | Gstab | `exp_Gstab_grad_stability.py` | `data/aggregated/v7_v1_exp_Gstab_grad/` | C32–C33 |
| Experiment 4.4 | snapshot spectral evolution | C2/E3 | `exp_C2_converged_sa.py`, `exp_E3_family_sa.py` | `data/aggregated/v7_v1_exp_C2_converged_sa/`, `data/aggregated/v7_v1_exp_E3_family_sa/`, `data/aggregated/v7_v1_exp_E3b_fair/` | C34–C35 |
| Experiment 5.1 | cross-architecture directional self-check | R1 | `exp_R1_arch_robust.py` | `data/aggregated/v7_v1_exp_R1_arch/` | C15, C57–C59 |
| Experiment 5.2 | two-reference-frame alignment probe | G3 | `exp_G3_c3prime_probe.py` | `data/aggregated/v7_v1_exp_G3_c3prime/` | C40–C44, C57–C59 |
| Experiment 5.3 | basin-entry paired test | G1b | `exp_G1b_pentry.py` | `data/aggregated/v7_v1_exp_G1b_pentry/` | C04–C10, C60–C61, C83 |
| Experiment 5.4 | in-basin strength regime | E1 | `exp_E1_beta_nu.py` | `data/aggregated/v7_v1_exp_E1_beta_nu/` | C62–C64, C68–C69 |
| Experiment 5.5 | training-budget scan (convergence robustness) | R2 | `exp_R2_budget_converge.py` | `data/aggregated/v7_v1_exp_R2_budget/` | C63 |
| Experiment 5.6 | degeneracy–bandwidth scan | E1b | `exp_E1b_smooth_degenerate.py` | `data/aggregated/v7_v1_exp_E1b_smooth/` | C65–C67 |
| Experiment 5.7 | clip-switch factorial | G1c | `exp_V2_G1c_clip_factor.py` | `data/aggregated/v7_v2_exp_V2_G1c_clip_factor/` | C45–C48, C56 |
| Experiment 5.8 | in-basin lr margin | B1b | `exp_V2_GstabB_stabilizer.py` | `data/aggregated/v7_v2_exp_V2_GstabB_stabilizer/` | C49–C52 |
| Experiment 5.9 | fixed-step L-BFGS control | B2b (includes the GC gradient-curve observation) | `exp_V2_GstabB_stabilizer.py` | same as above | C53–C55 |
| Experiment 6.1 | ground-truth-free basin discrimination | G2 | `exp_G2_basin_diagnose.py` | `data/aggregated/v7_v1_exp_G2_diagnose/` | C11–C13, C70–C82, C84 |
| Experiment 6.2 | threshold robustness scan | E-new-3 | `exp_Enew3_tau_scan.py` | `data/aggregated/v7_v1_exp_Enew3_tau_scan/` | C81 |
| Experiment 7.1 | smooth-Poisson degeneracy control | S2a | `exp_V2_S2a_poisson_degenerate.py` | `data/aggregated/v7_v2_exp_V2_S2a_poisson_degenerate/` | C85–C90 |
| Experiment 7.2 | internal turning-point layer | S2d | `exp_V2_S2d_internal_layer.py`, `exp_V2_S2d_main_accuracy.py`, `exp_V2_S2d_beta_effect.py`, `exp_V2_S2d_R_spectrum.py` | `data/aggregated/v7_v2_exp_V2_S2d_*/` | C91–C93 |
| Experiment 7.3 | Allen–Cahn internal spike | S2e | `exp_V2_S2e_spike.py`, `exp_V2_S2e_supervised.py` | `data/aggregated/v7_v2_exp_V2_S2e_*/` | C94–C96 |
| Experiment 7.4 | boundary-hugging convection–diffusion layer | S2c | `exp_V2_S2c_aligned_twostage.py`, `diag_S2c_lift.py`, `diag_S2c_stretch.py`, `diag_S2c_pos_basin.py` | `data/aggregated/v7_v2_exp_V2_S2c_aligned_twostage/` | C97–C98 |
| Experiment 7.5 | 1D time-dependent Burgers scan | D1 | `exp_D1_rational_beta_scan.py` | `data/aggregated/v8_time_varying_d1r/` | C99–C104 |
| Experiment 7.6 | artificial bad-basin escape diagnosis | caseB, s8 (two codenames for the same experiment) | `caseB_zeroIC_burgers.py`, `exp_S8_zeroIC_weight_clip.py`, `exp_B1B2_zeroIC_lbfgs.py` | `data/aggregated/v8_time_varying/` (canonical seeds 200–210) | C105 |
| Experiment 7.7 | 2D linear convection–diffusion | C4/C5 | `exp_C4_advdiff2d_baseline.py`, `exp_C5_advdiff2d_beta_scan.py` | `data/aggregated/v8_burgers_2d/` | C106–C107 |
| Experiment 7.8 | bandwidth–viscosity scan (for the record) | S1 (for-the-record experiment) | `exp_V2_S1_sigma_nu_scan.py` | `data/aggregated/v7_v2_exp_V2_S1_sigma_nu_scan/` | C113–C114 |

## Appendix A: intervention types (the three representation-layer intervention types, called Type-I/II/III in the paper; the S1/S2/S3 codenames are not used there)

| Repo codename | Type name | Content | Demonstrating experiment |
|---|---|---|---|
| S1 (intervention) | Type-I: boundary encoding | modifies $A$/$B$ (lifting and annihilation factors) | Experiment 7.4 (boundary-hugging layer) |
| S2 (intervention) | Type-II: coordinates and measure | modifies $X$ (coordinate map) | Experiment 7.2 (internal turning-point layer) |
| S3 (intervention) | Type-III: internal topology and features | modulates $\gamma$ and the initial value via defect location/topology | Experiment 7.3 (Allen–Cahn spike) |

Note: item (3) of §8.4, "matching representation bandwidth to solution scale", does not coincide exactly with the S3 definition in §7.2 (internal topology) — bandwidth belongs to the $\gamma$-feature side; the paper uniformly uses the phrasing "Type-III (feature/topology side)".

## Appendix B: script-level entries without experiment numbers

| Name | Codename | Note |
|---|---|---|
| Three-claims frozen self-check script | Vverify (script `exp_Vverify_claims.py`) | deterministic self-check at frozen points, not an independent experiment; referred to in the text as "frozen-point self-check (script in Appendix B)" |
| Four-point synthetic probe | toy_fourpoint(n=400) (function inside `exp_G2_basin_diagnose.py`) | inline synthetic illustration for the proposition in §6.2, not an independent experiment |
| Gradient-curve observation | GC | merged into Experiment 5.9 (Gstab-B combination), not listed separately |

## Notes on repo codenames

1. The codename "S1" has two meanings in the repo: an intervention type (boundary encoding) and the bandwidth–viscosity scan experiment (the "for the record" row). In the paper, the former is "Type-I" and the latter is Experiment 7.8.
2. "caseB" and "s8" are two codenames for the same diagnostic experiment, which is Experiment 7.6 in the paper.
3. The letter order of codenames S2c/S2d/S2e does not correspond to the intervention types they demonstrate (S2c demonstrates Type-I, S2d Type-II, S2e Type-III); the paper numbers (7.4/7.2/7.3) follow order of appearance in the text and are unrelated to the type ordering.
