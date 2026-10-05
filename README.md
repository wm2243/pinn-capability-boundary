# What Spatial Weighting Can and Cannot Change in PINNs

Spectral capability bounds of local diagonal weights + controlled empirical falsification: solution-set invariance, condition-number bounds, basin-selection falsification (1D viscous Burgers shock as the main test case).

## Papers

- `paper/paper1_v1_20_en.pdf` — English version (83 pages)
- `paper/paper1_v1_20.pdf` — Chinese version (77 pages)
- LaTeX sources in the same directory (compile with xelatex; handwritten `thebibliography`, no bibtex dependency)

## Links

- GitHub (this repository): https://github.com/wm2243/pinn-capability-boundary
- Zenodo full archive (including 144MB raw run data): DOI `10.5281/zenodo.23168250`
- HAL preprint: `[TBD]`

## Main claims

Organized by Taylor-expansion order, the effect of static spatial diagonal weighting is split into three levels (zeroth-order solution set / first-order gradient dynamics / second-order frozen-linearization spectrum), and its capability boundary is characterized level by level:

- **Second order (spectrum)**: local diagonal weights cannot change the correlation matrix $R(K_r)$ (Theorem 4.1); the condition number is already optimal on equicorrelation classes (Theorem 4.2, verified to exact equality in synthetic Experiment 4.1); on the real Burgers frozen Jacobian, Jacobi equilibration already closes >85% of the gap, leaving only 13%–24% for the unconstrained optimum (Experiments 4.2/4.3/4.4, static + in-training validation)
- **First order (dynamics)**: no evidence that weighting improves basin entry (Experiment 5.3, n=80 paired McNemar; uniform hit rate .613 above all weighted families); beneficial effects are confined to convergence-speed modulation inside an already-entered refinable basin (Mechanism A, Experiments 5.4/5.5/5.6) and gradient-scale modulation (down-weighting true localization, Experiments 4.3/5.7/5.8/5.9)
- **Methodology**: residual/loss self-assessment scalars with a fixed single projection cannot reliably discriminate good from bad basins (Proposition 6.1 + four-point counterexample; Experiment 6.1 validates the feasibility of the structural quantity $g_{et}$, proxy AUC .971 / oracle AUC .64)
- **Cross-equation validation**: no systematic benefit from up-weighting on smooth Poisson (Experiment 7.1); internal-singularity representation-layer defects dominate (Experiments 7.2/7.3/7.4); high-intensity up-weighting is near-neutral on time-dependent Burgers (Experiments 7.5/7.6); weighting is near-neutral on 2D linear convection–diffusion (Experiment 7.7)

## Repository layout

```
paper1_release/
├── README.md                  # this file
├── CITATION.cff               # citation metadata
├── LICENSE-MIT                # license for code
├── LICENSE-CC-BY-4.0          # license for paper/figures/data
├── EXPERIMENT_REGISTRY.md     # paper experiment numbers <-> repo codenames/scripts/data dirs
├── claims_provenance.csv      # provenance ledger for 115 quantitative claims
├── requirements.txt
├── paper/                     # papers (PDF + TeX sources + figures + provenance tables)
├── code/
│   ├── v7_1d/                 # 1D steady Burgers experiments
│   └── v8_2d/                 # time-dependent and 2D experiments
├── analysis/                  # figure/table generation scripts for the paper
├── tools/
│   ├── verify_claims.py       # automated claim checking (22 checks)
│   └── gen_provenance.py      # provenance-table generator (CN/EN)
├── data/
│   ├── aggregated/            # aggregated statistics CSVs (source of paper figures/tables)
│   └── raw/                   # raw run records (GitHub carries only the Phase0 cache; full data on Zenodo)
└── figures/                   # figures in both languages (pdf+png)
```

## Reproduction

### Environment
- Python 3.10+, dependencies in `requirements.txt` (torch / numpy / scipy / matplotlib / pandas; mpmath optional)
- Optional: DeepXDE (PyTorch backend) — only for the time-dependent control scripts of Experiments 7.5/7.6; no sklearn dependency anywhere
- Seed set: 11 equally spaced base seeds {0,5,...,50}

### Building the papers
```bash
cd paper
xelatex -interaction=nonstopmode paper1_v1_20_en.tex   # English version (run twice)
xelatex -interaction=nonstopmode paper1_v1_20.tex      # Chinese version (run twice)
```

### Checking quantitative claims
```bash
python tools/verify_claims.py    # 22 checks: CSV structure, provenance-table sync, recomputation of 12 key numbers
```

### Running experiments
- The mapping between paper experiment numbers and repo codenames/scripts/data directories is in `EXPERIMENT_REGISTRY.md`
- The Phase0 basin cache ships with the repo at `data/raw/v7_v1/_gate_cache/` (394 .pt files), so a fresh clone does not need to retrain Phase0
- Full raw run data (144MB) is in the Zenodo archive; aggregated CSVs ship with the repo
- Runtime outputs go to `code/v7_1d/results/` (gitignored)

## Honesty statements

- The contribution of this work is **characterizing capability boundaries with controlled falsification**, not proposing a new algorithm
- The post-hoc optimal β* is a descriptive observation (single-seed spread 1–100, non-monotone in ν) and **does not constitute a fixed strategy that can be deployed before training**
- The time-dependent/2D cases are lightweight cross-equation validations (n=11); hyperbolic systems such as water hammer and shallow water are left to future work
- The hard-boundary encoding amplitude bug (Appendix B.5) has been audited: zero impact for ν≤.01; .3/.5/1 rerun and archived
- N=1024 is excluded from quantitative conclusions because κ(R)≳1e13 approaches the double-precision limit

## Licenses

- Code (`code/`, `analysis/`, `tools/`): MIT (see `LICENSE-MIT`)
- Paper, figures, data (`paper/`, `figures/`, `data/`): CC BY 4.0 (see `LICENSE-CC-BY-4.0`)

## Citation

```bibtex
@misc{pinn_weighting_boundary_2026,
  title  = {What Spatial Weighting Can and Cannot Change in PINNs:
            Spectral Capability Bounds of Local Diagonal Weights with Controlled Falsification},
  author = {Wen, Ning},
  year   = {2026},
  howpublished = {HAL preprint [ID TBD]; code and data: Zenodo DOI 10.5281/zenodo.23168250}
}
```
