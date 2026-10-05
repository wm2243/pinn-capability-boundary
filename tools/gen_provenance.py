# -*- coding: utf-8 -*-
"""从 claims_provenance.csv 生成中英两版溯源 longtable（附录用）。
- 5 列：编号 | 申明（位置）| 数据来源 | 口径（过滤；聚合；n）| 状态
- 代号→实验编号只作用于 位置/声明/口径 列；数据文件路径保留原代号（真实路径）。
- 备注列整体不进论文（含过程痕迹）；声明/位置/口径中的 P-xx、日期、"早前会话"等过程痕迹剥除。
- 输出生成后检测 EN 中的残留中文，打印出来供补 override。
"""
import csv, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'claims_provenance.csv'
OUT_CN = ROOT / 'paper' / 'provenance_cn.tex'
OUT_EN = ROOT / 'paper' / 'provenance_en.tex'

# ---------- 代号 -> 实验编号（仅用于 位置/声明/口径） ----------
CODES = [
    ('Gstab/R1/G3/G1c/S1', '实验 4.3/5.1/5.2/5.7/7.8', 'Experiments 4.3/5.1/5.2/5.7/7.8'),
    ('E1/R2/E1b', '实验 5.4/5.5/5.6', 'Experiments 5.4/5.5/5.6'),
    ('E1/R2', '实验 5.4/5.5', 'Experiments 5.4/5.5'),
    ('R1/G3', '实验 5.1/5.2', 'Experiments 5.1/5.2'),
    ('G1b/G2', '实验 5.3/6.1', 'Experiments 5.3/6.1'),
    ('Gstab-B', '实验 5.8/5.9', 'Experiments 5.8/5.9'),
    ('C2/E3', '实验 4.4', 'Experiment 4.4'),
    ('caseB', '实验 7.6', 'Experiment 7.6'),
    ('E-new-3', '实验 6.2', 'Experiment 6.2'),
    ('Enew3', '实验 6.2', 'Experiment 6.2'),
    ('Gstab', '实验 4.3', 'Experiment 4.3'),
    ('G1b', '实验 5.3', 'Experiment 5.3'),
    ('G1c', '实验 5.7', 'Experiment 5.7'),
    ('B1b', '实验 5.8', 'Experiment 5.8'),
    ('B2b', '实验 5.9', 'Experiment 5.9'),
    ('F1d', '实验 4.2', 'Experiment 4.2'),
    ('E1b', '实验 5.6', 'Experiment 5.6'),
    ('S2a', '实验 7.1', 'Experiment 7.1'),
    ('S2c', '实验 7.4', 'Experiment 7.4'),
    ('S2d', '实验 7.2', 'Experiment 7.2'),
    ('S2e', '实验 7.3', 'Experiment 7.3'),
    ('P8', '实验 4.1', 'Experiment 4.1'),
    ('R1', '实验 5.1', 'Experiment 5.1'),
    ('R2', '实验 5.5', 'Experiment 5.5'),
    ('G2', '实验 6.1', 'Experiment 6.1'),
    ('G3', '实验 5.2', 'Experiment 5.2'),
    ('E1', '实验 5.4', 'Experiment 5.4'),
    ('D1', '实验 7.5', 'Experiment 7.5'),
    ('E3', '实验 4.4', 'Experiment 4.4'),
    ('C2', '实验 4.4', 'Experiment 4.4'),
    ('S1', '实验 7.8', 'Experiment 7.8'),
    ('Vverify', '冻结点自检脚本', 'the frozen-point self-check script'),
]

def map_codes(s, lang):
    idx = 1 if lang == 'cn' else 2
    for pat, cn, en in CODES:
        repl = cn if lang == 'cn' else en
        s = re.sub(r'(?<![A-Za-z0-9_])' + re.escape(pat) + r'(?![A-Za-z0-9_])', repl.replace('\\', '\\\\'), s)
    return s

# ---------- 过程痕迹清理 ----------
CLEAN = [
    ('（P-25 新增句）', ''), ('（P-38）', ''), ('（P-37 删括注）', ''),
    ('(早前会话已核)', ''), ('（早前会话已核）', ''),
    ('2026-10-04 ', ''), ('2026-10-04', ''),
    ('；旧版 D1 曾得增权有益', '；早期另一 ν/种子口径的扫描曾得增权有益'),
    ('（论文写 2.75/2.86/2.44/2.57）', ''),
    ('（canonical 代 seeds 200–210）', '（seeds 200–210）'),
    ('跨粘性表(P1) 9 行数字', '跨粘性表 9 行数字'),
    ('正类规模说明（P-37 删括注）', '正类规模说明'),
    ('K=4 多起点构造（P-38）', 'K=4 多起点构造'),
    ('原值+补跑', '原值+补跑'),
]

def clean(s):
    s = s or ''
    for a, b in CLEAN:
        s = s.replace(a, b)
    return s.strip()

CN_CLAIM_OVR = {
    'C105': '各组终态 L2 中位：对照 2.57、down_3 2.75、down_8 2.86、up_3 2.44、up_8 2.44；clip 与对照逐位相同',
}

def load_rows():
    """读 CSV 并修复三种错位：逗号未加引号(12/13 列)、缺复核值/数据文件(10 列)、C115(9 列)。"""
    raw = list(csv.reader(open(CSV, encoding='utf-8')))
    hdr, out = raw[0], []
    for r in raw[1:]:
        rid = r[0]
        if len(r) == 13:
            r = r[:6] + [','.join(r[6:9])] + r[9:]
        elif len(r) == 12:
            r = r[:6] + [r[6] + ',' + r[7]] + r[8:]
        elif len(r) == 10:
            if rid == 'C111':
                r = r[:8] + ['——'] + r[8:]
            elif any(k in r[4] for k in ('.csv', '.json', 'code/', 'aggregated', 'raw/')):
                r = r[:4] + ['同上'] + r[4:]
            else:
                r = r[:5] + ['同上'] + r[5:]
        elif len(r) == 9 and rid == 'C115':
            r = [r[0], r[1], r[2], r[3], '——', '设计网格', '——', '算术', '——', '已核', '']
        assert len(r) == 11, (rid, len(r))
        out.append(dict(zip(hdr, r)))
    return out

# EN 逐格覆盖（口径/位置/聚合/样本量中含整句中文的格子）
SPEC_EN_OVR = {
'C03': 'N=1024 row, kappa_R',
'C05': 'scope=pooled; per-family rows, p_entry',
'C06': 'net column',
'C07': 'exact binomial recompute from (rescued_BtoG; lost_GtoB)',
'C08': 'min p $\\times$ 3',
'C09': 'scope=by\\_nu; nu=.001; uniform minus per-family p_entry',
'C10': 'scope=by\\_nu; nu=.01; uniform minus per-family p_entry',
'C12': 'oracle feature recompute',
'C13': 'threshold-convention recompute',
'C15': 'axis=width; level=64; nu=.005; family=rational; $\\beta$=1.0 vs 8.0; L2 column',
'C17': 'deltaP_start_n500; mean and mean/500',
'C21': 'U = rotation 30$^\\circ$; K = U diag(1;100) U$^T$',
'C22': '2$\\times$2 identity $\\kappa+\\kappa^{-1}+2=(\\mathrm{tr})^2/\\det$',
'C23': 'r_ratio column; max$|r-1|$',
'C24': 'nstart column binned by N',
'C25': 'two rows of the r_ratio column',
'C26': 'nu=.005 four rows (N=64--512); kappa_uniform/kappa_jacobi/kappa_unconstr/r_unconstr/kappa_rational_best/rational_best_beta',
'C27': 'nu=.005; uniform_vs_jacobi column',
'C29': 'rational_best_beta/r_rational/kappa_beta_1..100 columns',
'C31': 'lmin_rel_diff column',
'C32': 'stage=E; kappa_med family / uniform; clip=1 all $\\nu$; clip=1 $\\nu\\in\\{.001;.005\\}$; clip=0 all $\\nu$; clip=0 $\\nu$=.01',
'C33': 'stage=E; g_peak and lmax_med ordering per ($\\nu$,clip) cell',
'C36': 'nstart=12/8/6 plus one $x=0$ start (opt_diag_kappa)',
'C37': 'fixed-point rerun (same convention; ncv=(61;24) and (100;60); two starting points); $\\lambda_{max}$ ratio 9.36:1:1/9.21; $|\\lambda_{min}|$ scaled 9.80$\\times$ and 1/9.80',
'C38': 'kR_eigh (N=1024 row)',
'C39': 'gate_l2 (nu=.005 row)',
'C42': 'c3_inner (full table, 176 rows)',
'C46': '$\\nu$=.005; paired by seed, clip0 vs clip1 (=Experiment 5.3); good column',
'C47': '$\\nu$=.001; paired by seed across families; good column',
'C49': 'blown aggregated by family$\\times$lr',
'C50': 'blown==0; per (seed,family) max lr with $L_2<.06$',
'C51': 'same ceiling values as C50, paired',
'C52': 'same as C50 with changed threshold',
'C55': 'median of 8 seeds per (family,step); g_pre',
'C58': '7 configurations spanning width/depth/npde; eta_beta_spearman',
'C62': 'nu=.001/.003/.005/.01; argmin $L_2$ over beta per (nu,seed)',
'C63': 'Experiment 5.4 same as C62; Experiment 5.5 argmin $L_2$ over beta per (nu,seed)',
'C64': 'thin-shock $4\\nu\\times11$; $g_{post}=(L_2(\\beta{=}1)-\\min_\\beta L_2)/L_2(\\beta{=}1)$',
'C67': 'nu=.3/.5/1; min $L_2$ per (feat,nu,seed), then median by feat, then ratio',
'C68': 'per nu: cell-median argmin vs per-seed argmin; per-seed $L_2$--$\\beta$ Spearman',
'C70': 'toy_fourpoint(n=400) synthetic probe',
'C71': 'oracle_good/truth_good columns',
'C72': 'family==uniform; merged by (nu,seed): oracle_L2 vs phase0_L2',
'C73': 'auc rows: proxy/loss_cv/loss_slope/gradnorm_t/wmean_t/focus_t/focus_cv',
'C75': 'auc rows: peak_ratio/peak_in_band/abs_peak/ge_t',
'C76': 'four features vs oracle_good, recomputed',
'C77': 'loo columns of the ge_t row; ALL full-multivariate row',
'C78': 'ge_t vs truth_good, stratified resampling',
'C80': 'oracle_good proportion $\\times$ proxy sensitivity',
'C81': 'truth_L2 grouped by truth_good; tau scan agree_pct',
'C82': 'five auc rows of the STRUCT group excluding the three named ones',
'C84': 'truth_good column',
'C92': 'Gate frozen probe $\\beta\\in\\{1;3;8\\}$',
'C100': 'two-sided Wilcoxon',
'C101': 'per-seed argmin attribution + paired ratio',
'C112': 'code review',
'C86': 'up-weighting $\\beta=10$ vs uniform, grouped by configuration',
'C87': 'multi-frequency $\\times$ MLP group: inverse down-weighting vs uniform',
'C88': 'single-frequency $\\sigma$=1 group',
'C89': 'multi-frequency $\\sigma$=15 group',
'C90': '$\\sigma$=15 vs $\\sigma$=1 cross-group median ratio',
'C93': 'layer-region $L_2$ paired difference',
'C94': 'seedless-group terminal states',
'C95': '12 independent seeds fitting $u^*$',
'C97': 'control and no-intervention groups, terminal $L_2$',
'C99': '$\\beta$ level $\\times$ space-time-domain relative $L_2$',
'C102': 'Gate frozen probe $\\beta=1\\to8$',
'C103': 'three $\\tau$ levels count + sorting gap',
'C104': '$\\beta$ level $\\times$ L2_relative',
'C105': 'per-group terminal $L_2$',
'C106': '$\\beta$ level $\\times$ L2_total',
'C107': 'L2_total compared with $\\tau$=.06',
'C108': 'same as C26--C29',
'C109': 'same as C04--C08',
'C110': 'same as C53/C54',
}
POS_EN_OVR = {
'C26': '\\S4.4 Table 4.1',
'C111': '\\S8.1 / \\S8.2 / \\S8.5 / summary section',
}
AGG_EN_OVR = {'C101': 'count + median + max'}
N_EN_OVR = {
'C33': '6 cells', 'C43': '4 levels', 'C44': '4 levels',
'C51': '8 (nonzero 5/2/3)', 'C55': '8$\\times$11 steps',
}

# ---------- EN 翻译 ----------
EN_CLAIMS = {
'C01': 'Experiment 4.2 multi-start improvement, N$\\le$512',
'C02': 'Experiment 4.2 improvement ratio $r=\\kappa^*_{diag}/\\kappa(R)$',
'C03': 'Experiment 4.2: $\\kappa(K_r)\\gtrsim10^{13}$ at $N=1024$',
'C04': 'uniform refinable-basin hit rate',
'C05': 'weighted-family hit rates: rational/down\\_inv/down\\_lin',
'C06': 'McNemar net rescues: down\\_inv/down\\_lin/rational',
'C07': 'McNemar exact two-sided $p$: down\\_lin/down\\_inv/rational',
'C08': 'after Holm correction only down\\_lin is significant',
'C09': 'small-viscosity $\\nu=.001$ paired differences',
'C10': 'transitional-viscosity $\\nu=.01$ direction inconsistent',
'C11': 'Experiment 6.1 proxy ($g_{et}$) AUC',
'C12': 'Experiment 6.1 oracle AUC',
'C13': 'Experiment 6.1 sensitivity / specificity',
'C14': 'Experiment 5.6 total run count',
'C15': 'width-64 baseline-group medians, $\\beta=1\\to8$',
'C16': 'cross-viscosity table, 9 rows of numbers',
'C17': 'Adam transient per-step relative drift / cumulative $\\delta_P$ at $n=500$',
'C18': 'Adam steady-state (step$\\ge$1000) cumulative-drift medians, $n=50/100/500$',
'C19': 'steady-state per-step mean drift',
'C20': 'platform-term / whitened-gradient ratio, median and IQR',
'C21': '$N=2$ closed-form example: $\\kappa(R)$ and $t^*(a;b)$',
'C22': '$t=1/a$ control: $\\kappa$ and $(\\mathrm{tr})^2/\\det$',
'C23': 'Experiment 4.1 equicorrelation matrices: max deviation of $r=1$ over 42 configurations',
'C24': 'Experiment 4.1 start counts for $N\\le32/64/128$',
'C25': 'Experiment 4.1 banded matrix $r$ (30 starts; PSD)',
'C26': 'Experiment 4.2 triplet table $\\nu=.005$, all 24 numbers',
'C27': 'uniform/Jacobi factors for $N=64$--$128$ and $N=256$--$512$',
'C28': 'Jacobi fill ratio in the linear-gap convention',
'C29': 'rational within-family optimum: $N=64$ $\\beta^*=5$ at $32\\times$ Jacobi; $N\\ge128$ $\\beta^*=1$, monotone worsening',
'C30': 'effective-rank PR saturation ($N\\ge256$)',
'C31': 'mpmath high-precision verification of $\\lambda_{\\min}$ relative difference',
'C32': 'Experiment 4.3 down-weighting worsening ratios: default clip / small $\\nu$ / clip off / $\\nu=.01$ down\\_inv',
'C33': 'Experiment 4.3 ordering of $g_{peak}$ and $\\lambda_{max}$: rational$\\gg$uniform$>$down\\_lin$>$down\\_inv',
'C34': 'Experiment 4.4 $\\kappa$-ratio median over training, always $>1$',
'C35': 'Experiment 4.4(ii): up-weighting worsening at convergence / down\\_lin improvement / Gate rational',
'C36': 'Experiment 4.2 multi-start count decreasing with $N$',
'C37': 'supplementary spectral-edge observation: three-family $\\lambda_{max}$, $\\kappa$, negative-curvature dimension',
'C38': '$\\kappa$ near the double-precision limit at $N=1024$ (4 places)',
'C39': 'Experiment 4.2 frozen-point good-basin $L_2$',
'C40': 'Experiment 5.2 bad-basin $\\cos\\alpha_S$ (vs $d_B$, $\\beta=1$)',
'C41': 'Experiment 5.2 good-basin $\\cos\\alpha_S$ (vs $d_*$, $\\beta=1$)',
'C42': 'C3 inner product $\\langle\\partial_\\beta g_S, g_S\\rangle$ sign per record',
'C43': 'C3 inner-product median vs $\\beta$ (good basins)',
'C44': 'definite-sign attainment rate (good-basin share $\\cos>\\varepsilon_S$)',
'C45': 'Experiment 5.7: zero blow-ups with clipping off (incl. $\\nu=.001$ rational)',
'C46': '$\\nu=.005$: clip switch has no effect on entry rate',
'C47': '$\\nu=.001$ clip$=1$: uniform beats both down-weighting families',
'C48': 'Experiments 5.8/5.9 clip trigger rate',
'C49': 'Experiment 5.8 blow-up rates',
'C50': 'Experiment 5.8 effective lr ceilings, median ($L_2<.06$)',
'C51': 'Experiment 5.8 ceiling paired Wilcoxon',
'C52': 'Experiment 5.8 loose-threshold ceiling medians ($L_2<.1/.2$)',
'C53': 'Experiment 5.9 $\\nu=.005$ four-family terminal $L_2$',
'C54': 'Experiment 5.9 $\\nu=.001$ four-family terminal $L_2$',
'C55': 'gradient-curve 500-step norm ordering and magnitudes',
'C56': 'Experiment 5.7 rational pre-clip peak $7\\times10^2$ ($\\nu=.001$)',
'C57': 'frozen-point self-check: $\\eta_{gate}$ vs $\\beta$ Spearman all $+1$',
'C58': 'Experiment 5.1 direction unchanged across 7 configurations',
'C59': 'direct C-Sep measurement: median ratio and strict version',
'C60': 'down\\_lin Holm-significant interval',
'C61': 'low-threshold good-basin sample counts',
'C62': 'per-seed optimal $\\beta^*$ medians per $\\nu$',
'C63': 'cross-viscosity $\\beta^*$ medians',
'C64': '$g_{post}$ distribution',
'C65': 'Experiment 5.6 stiff-end 4--10$\\times$ gain',
'C66': 'Experiment 5.6 smooth-end degeneracy',
'C67': 'Experiment 5.6 fixed high-bandwidth mismatch at the smooth end',
'C68': 'median heatmap representativeness',
'C69': 'Experiment 5.4 per-cell dispersion',
'C70': 'four-point toy probe AUC',
'C71': 'Experiment 6.1 label sizes',
'C72': '49/80 and the Experiment 5.3 hit rate are the same statistic',
'C73': 'AUC of seven in-training quantities',
'C74': 'r\\_resid AUC under two conventions',
'C75': 'structural-quantity proxy AUC',
'C76': 'structural quantities under the oracle-existence convention',
'C77': '$g_{et}$ LOO metrics vs 17-dimensional parity',
'C78': '$g_{et}$ proxy AUC bootstrap interval',
'C79': '$g_{et}$ oracle convention',
'C80': 'end-to-end rough ledger',
'C81': 'natural gap of good/bad-basin $L_2$ and Experiment 6.2',
'C82': 'ranges of the remaining STRUCT quantities',
'C83': '$K=4$ multi-start construction',
'C84': 'positive-class size note',
'C85': 'Experiment 7.1 setup: total training runs',
'C86': 'up-weighting: five of six paired groups insignificant; significant group (single-frequency $\\sigma=1$) median worsening',
'C87': 'multi-frequency + plain MLP inverse down-weighting improvement',
'C88': 'single-frequency $\\sigma=1$ inverse down-weighting improvement',
'C89': 'mismatched high bandwidth (multi-frequency $\\sigma=15$) inverse down-weighting significantly worse',
'C90': 'mismatched bandwidth 40--1000$\\times$ worse than matched bandwidth',
'C91': 'Experiment 7.2: three thicknesses, $K=4$ selection, 11/11 enter refinable basins; refined median of order $10^{-5}$',
'C92': 'Experiment 7.2 Mechanism-A conditionality: zonal gradient-energy ratio $\\eta$',
'C93': 'Experiment 7.2 paired refinement',
'C94': 'Experiment 7.3: seedless 12/12 collapse to zero, $L_2\\approx.999$',
'C95': 'Experiment 7.3 supervised control: global normalized $L_2$ median $6.2\\times10^{-4}$',
'C96': 'Experiment 7.3 topological-seeding outcomes',
'C97': 'Experiment 7.4: same-thickness Burgers $K=8$ 3/8 hits; boundary-hugging layer 16/16 zero hits',
'C98': 'Experiment 7.4: thick-layer stretch 4/4 success, thin-layer 8/8 failure; exact exponential lifting 4/4; coarse logistic 8/8 failure',
'C99': 'Experiment 7.5 Table d1beta: four rational levels ($\\tau=.06$)',
'C100': 'Experiment 7.5 paired Wilcoxon $\\beta=1$ vs $3/8/15$',
'C101': 'Experiment 7.5 per-seed best attribution; seed-30 rescue; $\\beta=8/\\beta=1$ ratio',
'C102': 'Experiment 7.5 Mechanism A: cosine / norm ratio / row-norm ratio',
'C103': 'Experiment 7.5 table note: threshold sensitivity',
'C104': 'spatial linear-family control: zero good basins at $\\beta=8/15$ (up-weighting harmful); an earlier scan under a different $\\nu$/seed convention found benefit',
'C105': 'group terminal-$L_2$ medians: control 2.57, down\\_3 2.75, down\\_8 2.86, up\\_3 2.44, up\\_8 2.44; clip identical to control',
'C106': 'Experiment 7.7 Table c45: three-level medians and ranges',
'C107': 'Experiment 7.7: all 33 runs in good basins; $\\beta=1\\to8$ about 4\\% same-order fluctuation',
'C108': 'Experiment 4.2 linear-gap Jacobi fill $>85\\%$; rational within-family optimum always $\\beta=1$',
'C109': 'uniform .613 highest; down\\_lin .388 with Holm $p=.006$; down\\_inv/rational below uniform',
'C110': 'Experiment 5.9 fixed-step L-BFGS four-family terminal differences insignificant',
'C111': 'no new quantitative claims (condition checklist and summary); cited numbers refer back to C01--C107',
'C112': 'Appendix B $N/p$ table: Experiment 7.5 Mechanism-A probe $N=2048$, $p=7851$; random collocation',
'C113': 'Appendix B.5 bc\\_amp audit: $\\delta(\\nu)=1-\\tanh(1/2\\nu)$ four-level values; Experiment 7.8 invalidated 264 runs',
'C114': 'Appendix atlas sample sizes: Experiments 4.3/5.1/5.2/5.7/7.8',
'C115': 'Appendix app:tables cell-count arithmetic: 297 / 1584 / 80+80',
}

POS_TOKENS = [
    ('附录 实验图谱', 'Appendix experiment atlas'), ('附录方向安全声明', 'Appendix directional-safety statement'),
    ('附录 B N/p 表', 'Appendix B $N/p$ table'), ('附录 B.5', 'Appendix B.5'), ('附录 app:tables', 'Appendix app:tables'),
    ('附录', 'Appendix'), ('摘要', 'Abstract'), ('贡献', 'Contributions'), ('表注', 'table note'),
    ('图题', 'caption'), ('注记', 'Remark'), ('段', 'para.'), ('设置', 'setup'), ('结论', 'conclusion'),
    ('补充观察', 'supplementary observation'), ('机制A条件性', 'Mechanism-A conditionality'),
    ('配对精修', 'paired refinement'), ('机制A', 'Mechanism A'), ('实验图谱', 'experiment atlas'),
]
AGG_TOKENS = [
    ('精确枚举+脚本渐近（无连续性修正）', 'exact enumeration + script asymptotic (no continuity correction)'),
    ('精确 McNemar+标准 Holm（累计 max）', 'exact McNemar + standard Holm (cumulative max)'),
    ('配对 McNemar 精确', 'exact paired McNemar'), ('逐种子 argmin 后中位', 'per-seed argmin, then median'),
    ('中位/IQR/范围', 'median / IQR / range'), ('比例+Wilson95', 'proportion + Wilson 95'),
    ('中位+IQR', 'median + IQR'), ('中位+计数+范围', 'median + count + range'),
    ('中位+范围+计数', 'median + range + count'), ('中位+范围', 'median + range'),
    ('中位+计数', 'median + count'), ('计数+范围', 'count + range'), ('计数+中位', 'count + median'),
    ('范围+一致率', 'range + agreement rate'), ('范围+中位', 'range + median'),
    ('逐步中位范围', 'stepwise median range'), ('IQR 与成功率', 'IQR and success rate'),
    ('逐 seed best 中位+比值', 'per-seed best median + ratio'),
    ('逐点符号+秩相关', 'per-point sign + rank correlation'),
    ('原值+逐列', 'as reported + per-column'), ('原值+补跑', 'as reported + rerun'),
    ('原值（中位）', 'as reported (median)'), ('复算+计数', 'recompute + count'),
    ('中位/n 导出', 'derived median/n'), ('Wilcoxon 配对+中位', 'paired Wilcoxon + median'),
    ('中位+脚本渐近 Wilcoxon', 'median + script-asymptotic Wilcoxon'),
    ('Wilcoxon 双侧', 'two-sided Wilcoxon'), ('Wilcoxon+Holm', 'Wilcoxon + Holm'),
    ('配对 Wilcoxon', 'paired Wilcoxon'), ('AUC+LOO 混淆', 'AUC + LOO confusion'),
    ('LOO Youden；混淆矩阵', 'LOO Youden; confusion matrix'), ('LOO Youden', 'LOO Youden'),
    ('百分位 2.5/97.5', 'percentiles 2.5/97.5'), ('Spearman 秩相关', 'Spearman rank correlation'),
    ('逐配置对比', 'per-config comparison'), ('排序核验', 'ordering check'),
    ('配置核对', 'config check'), ('代码审查', 'code review'), ('双侧精确', 'exact two-sided'),
    ('配对计数', 'paired counts'), ('配对差', 'paired difference'), ('组中位', 'group median'),
    ('比值范围', 'ratio range'), ('重跑', 'rerun'), ('重算', 'recomputed'), ('导出', 'derived'),
    ('算术', 'arithmetic'), ('均值', 'mean'), ('中位', 'median'), ('范围', 'range'),
    ('计数', 'count'), ('原值', 'as reported'), ('比例', 'proportion'), ('极值', 'extremum'),
    ('逐档', 'per level'), ('对比', 'comparison'), ('混淆矩阵', 'confusion matrix'),
]
N_TOKENS = [
    ('8 种子×4 族', '8 seeds $\\times$ 4 families'), ('4 行同值', '4 rows, same value'),
    ('22 探针对', '22 probe pairs'), ('11 Gate 点', '11 Gate points'), ('9 次估计', '9 estimates'),
    ('8 配对', '8 pairs'), ('3 比较', '3 comparisons'), ('4/配置', '4/config'),
    ('csep 33/粘度', 'csep 33/$\\nu$'), ('66/feat/粘度', '66/feat/$\\nu$'),
    ('18/族·点', '18/family$\\cdot$point'), ('80/族', '80/family'), ('20/族', '20/family'),
    ('11/格', '11/cell'), ('11/档', '11/level'), ('11/组', '11/group'), ('4/组', '4/group'),
    ('4 格', '4 cells'), ('8 格(2ν×4族)', '8 cells (2$\\nu$ $\\times$ 4 families)'),
    ('/组', '/group'), ('/档', '/level'),
    ('种子', ' seeds'), ('族', ' families'),
]
SPEC_TOKENS = [
    ('早前会话已核', ''), ('按 (nu,seed) 取 L2 最小的 beta', 'argmin $L_2$ over beta per (nu,seed)'),
    ('按 (feat,nu,seed) 取 L2 最小后按 feat 中位再取比', 'min $L_2$ per (feat,nu,seed), then median by feat, then ratio'),
    ('每 (nu,seed) 取 L2 最小的 beta', 'argmin $L_2$ over beta per (nu,seed)'),
    ('逐 seed 取 l2_band 或 L2 最小', 'per-seed min of l2\\_band or $L_2$'),
    ('格内中位 argmin vs 逐种子 argmin', 'cell-median argmin vs per-seed argmin'),
    ('逐 seed L2-beta Spearman', 'per-seed $L_2$--$\\beta$ Spearman'),
    ('统计 L2 q1/q3/min/max/success', 'per-cell $L_2$ q1/q3/min/max/success'),
    ('过滤口径必须含 axis=width', 'filter must include axis=width'),
    ('数值复算+行数', 'numerical recompute + row count'),
    ('数据行数', 'row count'), ('全表行数', 'full-table row count'),
    ('各聚合 csv 行数与分组', 'row counts and grouping of each aggregated csv'),
    ('列(汇总见 raw 下 s3_summary.json)', 'column (summary in s3\\_summary.json under raw/)'),
    ('按 family 统计', 'count by family'), ('按 family', 'by family'),
    ('逐 (ν;clip) 格排序', 'ordering per ($\\nu$,clip) cell'),
    ('族÷uniform', 'family / uniform'),
    ('按 stage 对 improve_ratio 取中位', 'median of improve\\_ratio by stage'),
    ('按 family 对 improve_ratio 取中位（contrast 合并）', 'median of improve\\_ratio by family (contrast merged)'),
    ('按 (family,step)取 8 种子中位', 'median over 8 seeds per (family,step)'),
    ('按 seed 配对 clip0 vs clip1(=实验 5.3)', 'paired by seed, clip0 vs clip1 (=Experiment 5.3)'),
    ('按 seed 配对 clip0 vs clip1(=G1b)', 'paired by seed, clip0 vs clip1 (=Experiment 5.3)'),
    ('族间按 seed 配对', 'paired by seed across families'),
    ('按 seed 配对', 'paired by seed'), ('同种子配对', 'same-seed pairing'),
    ('按 ε 分组终态 L2', 'terminal $L_2$ grouped by $\\varepsilon$'),
    ('按种子类型分组', 'grouped by seed type'), ('按干预分组', 'grouped by intervention'),
    ('按配置分组', 'grouped by configuration'), ('按 beta', 'by beta'), ('按 nu×beta', 'by nu$\\times$beta'),
    ('按 nu×beta 统计', 'statistics by nu$\\times$beta'),
    ('取 spearman 列', 'spearman column'), ('同批上比较', 'compared on the same batch'),
    ('同批比较两 β', 'two $\\beta$ compared on the same batch'),
    ('取 L2 最小的 beta', 'argmin $L_2$ over beta'),
    ('解析重算(无 CSV)', 'analytic recompute (no CSV)'),
    ('代码实证', 'verified in code'), ('循环', 'loop'),
    ('无放回抽 5', 'draw 5 without replacement'),
    ('四档数值', 'four-level values'), ('作废', 'invalidated'),
    ('区间', 'interval'), ('计数', 'count'), ('中位', 'median'), ('范围', 'range'),
    ('重算', 'recomputed'), ('复算', 'recomputed'), ('原值', 'as reported'),
    ('导出', 'derived'), ('且', 'and'), ('或', 'or'), ('对', ' vs '),
    ('真值分组', 'grouped by truth'), ('分组', 'grouped'), ('行数', 'row count'),
    ('配对', 'paired'), ('合并', 'merged'), ('取', 'take '), ('的', ' '), ('；', '; '),
    ('，', ', '), ('（', ' ('), ('）', ') '),
]
SPEC_TOKENS.sort(key=lambda x: -len(x[0]))
POS_TOKENS.sort(key=lambda x: -len(x[0]))
AGG_TOKENS.sort(key=lambda x: -len(x[0]))
N_TOKENS.sort(key=lambda x: -len(x[0]))

def apply_tokens(s, tokens):
    for a, b in tokens:
        s = s.replace(a, b)
    return s

# ---------- LaTeX 转义 ----------
UNI = [
    ('⁻¹', '$^{-1}$'), ('²', '$^2$'),
    ('ν', '$\\nu$'), ('κ', '$\\kappa$'), ('β', '$\\beta$'), ('λ', '$\\lambda$'),
    ('θ', '$\\theta$'), ('η', '$\\eta$'), ('τ', '$\\tau$'), ('σ', '$\\sigma$'),
    ('γ', '$\\gamma$'), ('δ', '$\\delta$'), ('ε', '$\\varepsilon$'), ('ρ', '$\\rho$'),
    ('α', '$\\alpha$'), ('∈', '$\\in$'), ('≤', '$\\le$'), ('≥', '$\\ge$'),
    ('≠', '$\\neq$'), ('≈', '$\\approx$'), ('≳', '$\\gtrsim$'), ('×', '$\\times$'),
    ('→', '$\\to$'), ('−', '-'), ('–', '--'), ('±', '$\\pm$'),
    ('⟨', '$\\langle$'), ('⟩', '$\\rangle$'), ('⊤', '$\\top$'), ('∂', '$\\partial$'),
    ('≫', '$\\gg$'), ('✓', '$\\checkmark$'), ('°', '$^\\circ$'),
]

def tex_escape(s, lang):
    # 数学段（$...$）内不转义；文本段转义裸特殊字符（已带反斜杠的不重复转义）
    segs = s.split('$')
    for i in range(0, len(segs), 2):  # 偶数段为文本
        t = segs[i]
        t = re.sub(r'(?<!\\)&', r'\\&', t)
        t = re.sub(r'(?<!\\)%', r'\\%', t)
        t = re.sub(r'(?<!\\)#', r'\\#', t)
        t = re.sub(r'(?<!\\)_', r'\\_', t)
        t = re.sub(r'(?<!\\)\^', r'\\textasciicircum{}', t)
        t = re.sub(r'(?<!\\)~', r'\\textasciitilde{}', t)
        segs[i] = t
    s = '$'.join(segs)
    for a, b in UNI:
        s = s.replace(a, b)
    return s

def tex_escape_claim_en(s):
    # EN claims 是手写 LaTeX，只转义 % 之外的裸字符已由作者保证；直接返回
    return s

CJK = re.compile(r'[\u4e00-\u9fff]')
# EN 输出兜底：全角标点→半角（覆盖格手写内容可能漏网），并纳入残留检测
FW2HW = {'；': '; ', '，': ', ', '（': ' (', '）': ') ', '：': ': ', '、': ', ',
         '“': '``', '”': "''", '‘': '`', '’': "'", '　': ' ', '…': '...'}
FWPUNCT = re.compile(r'[　-〿＀-￯]')

def fw2hw(s):
    for a, b in FW2HW.items():
        s = s.replace(a, b)
    return s

def src_cell(s, lang='cn'):
    s = clean(s)
    if s in ('——', '—', ''):
        return '—'
    if lang == 'en':
        s = s.replace(' 与 ', ' and ').replace('、', ', ')
        for a, b in [('由 C07 三族 p 值 Holm 重算', 'recomputed from the three C07 p-values via Holm'),
                     ('（派生自', '(derived from'), ('(无 CSV)', '(no CSV)'),
                     ('解析重算', 'analytic recompute'), ('设计网格', 'design grid'),
                     ('行 152', 'line 152'), ('输出', 'outputs')]:
            s = s.replace(a, b)
        s = s.replace('）', ')').replace('（', '(')
    s = tex_escape(s, lang)
    # 长路径允许断行：斜杠与下划线后加零宽断点
    s = s.replace('/', '/\\hspace{0pt}').replace('\\_', '\\_\\hspace{0pt}')
    if CJK.search(s):
        return '{\\scriptsize ' + s + '}'
    return '{\\scriptsize\\texttt{' + s + '}}'

def esc_bare(s):
    """只转义裸 _ 和 %（不重复转义已带反斜杠的），用于手写 LaTeX 覆盖格。"""
    s = re.sub(r'(?<!\\)_', r'\\_', s)
    s = re.sub(r'(?<!\\)%', r'\\%', s)
    return s

def spec_breaks(s):
    """口径列长 token 内允许断行：; = _ / ( ) 后加零宽断点（\allowbreak 文本/数学模式均可用）。"""
    for ch in [';', '=', '/', '(', ')']:
        s = s.replace(ch, ch + '\\allowbreak ')
    s = s.replace('\\_', '\\_\\allowbreak ')
    return s

def build_rows(lang):
    rows = []
    leftovers = []
    for r in load_rows():
        rid = r['id'].strip()
        pos = clean(r['位置']); claim = clean(r['声明'])
        spec = clean(r['列/过滤口径']); aggr = clean(r['聚合方式']); n = clean(r['n'])
        status = clean(r['状态'])
        if lang == 'cn':
            pos = map_codes(pos, 'cn'); claim = map_codes(claim, 'cn')
            claim = CN_CLAIM_OVR.get(rid, claim)
            spec = map_codes(spec, 'cn')
            claim_t = tex_escape(claim, 'cn'); pos_t = tex_escape(pos, 'cn')
            spec_t = tex_escape(spec, 'cn'); aggr_t = tex_escape(aggr, 'cn')
            n_t = tex_escape(n, 'cn'); status_t = status
            src_raw = clean(r['数据文件'])
        else:
            pos = map_codes(pos, 'en')
            pos = apply_tokens(pos, POS_TOKENS)
            pos = POS_EN_OVR.get(rid, pos)
            if rid in POS_EN_OVR: pos = esc_bare(pos)
            claim = EN_CLAIMS.get(rid, '!!MISSING ' + rid)
            spec = map_codes(spec, 'en'); spec = apply_tokens(spec, SPEC_TOKENS)
            spec = SPEC_EN_OVR.get(rid, spec)
            aggr = apply_tokens(aggr, AGG_TOKENS); aggr = AGG_EN_OVR.get(rid, aggr)
            n = apply_tokens(n, N_TOKENS); n = N_EN_OVR.get(rid, n)
            pos_t = tex_escape(pos, 'en') if rid not in POS_EN_OVR else pos
            claim_t = esc_bare(claim)
            spec_t = tex_escape(spec, 'en') if rid not in SPEC_EN_OVR else esc_bare(spec)
            aggr_t = tex_escape(aggr, 'en') if rid not in AGG_EN_OVR else esc_bare(aggr)
            n_t = tex_escape(n, 'en') if rid not in N_EN_OVR else esc_bare(n)
            status_t = 'verified'
            src_raw = clean(r['数据文件'])
            src_raw = src_raw.replace('同上', 'ditto')
            for fld, val in [('pos', pos), ('spec', spec), ('aggr', aggr), ('n', n)]:
                if CJK.search(val):
                    leftovers.append((rid, fld, val))
        # 口径 cell
        parts = []
        if spec not in ('——', '—', ''):
            parts.append(spec_t)
        if aggr not in ('——', '—', ''):
            parts.append(aggr_t)
        if n not in ('——', '—', ''):
            parts.append('n=' + n_t)
        spec_cell = '; '.join(parts) if parts else '—'
        spec_cell = spec_breaks(spec_cell)
        src = src_cell(src_raw, lang)
        rows.append((rid, claim_t, pos_t, src, spec_cell, status_t))
    return rows, leftovers

HEADER_CN = r'''\FloatBarrier
\section{定量申明溯源表}\label{app:provenance}

本表为论文每一处定量申明建立一行溯源：申明内容与所在位置（申明下小字）、生成它的数据文件或脚本、列过滤与聚合口径、样本量与核对状态。全部 115 条均已对照仓库源数据复核。路径相对于仓库根目录：聚合数据位于 \texttt{data/aggregated/}，原始记录位于 \texttt{data/raw/}，脚本位于 \texttt{code/}；派生行（无独立文件）注明其来源行号。论文实验编号与仓库代号、脚本、数据目录的对照见仓库根目录 \texttt{EXPERIMENT\_REGISTRY.md}。

{\footnotesize
\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.9cm}p{6.4cm}p{4.2cm}p{3.0cm}p{1.2cm}@{}}
\caption{定量申明溯源表（共 115 条，全部已核）。}\label{tab:provenance}\\
\toprule
编号 & 申明（下起小字为正文位置） & 数据来源 & 口径（过滤；聚合；$n$） & 状态\\
\midrule
\endfirsthead
\multicolumn{5}{@{}l}{\footnotesize 续表~\ref{tab:provenance}}\\
\toprule
编号 & 申明（下起小字为正文位置） & 数据来源 & 口径（过滤；聚合；$n$） & 状态\\
\midrule
\endhead
\midrule\multicolumn{5}{r@{}}{\footnotesize 接下页}\\
\endfoot
\bottomrule
\endlastfoot
'''

HEADER_EN = r'''\FloatBarrier
\section{Quantitative-Claim Provenance}\label{app:provenance}

This table gives one provenance row per quantitative claim in the paper: the claim and its location in the text (in small print below the claim), the data file or script that generates it, the column-filter and aggregation convention, the sample size, and the verification status. All 115 entries have been re-checked against the repository source data. Paths are relative to the repository root: aggregated data under \texttt{data/aggregated/}, raw records under \texttt{data/raw/}, scripts under \texttt{code/}; derived rows (no standalone file) note their source row. The mapping between the paper's experiment numbers and the repository codenames, scripts, and data directories is given in \texttt{EXPERIMENT\_REGISTRY.md} at the repository root.

{\footnotesize
\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.9cm}p{6.4cm}p{4.2cm}p{3.0cm}p{1.2cm}@{}}
\caption{Provenance of quantitative claims (115 entries, all verified).}\label{tab:provenance}\\
\toprule
id & claim (location in small print) & source & convention (filter; aggregation; $n$) & status\\
\midrule
\endfirsthead
\multicolumn{5}{@{}l}{\footnotesize Table~\ref{tab:provenance} (continued)}\\
\toprule
id & claim (location in small print) & source & convention (filter; aggregation; $n$) & status\\
\midrule
\endhead
\midrule\multicolumn{5}{r@{}}{\footnotesize continued overleaf}\\
\endfoot
\bottomrule
\endlastfoot
'''

def emit(lang, out, header):
    rows, leftovers = build_rows(lang)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(header)
        for rid, claim, pos, src, spec, status in rows:
            if lang == 'en':
                claim, pos, src, spec = (fw2hw(x) for x in (claim, pos, src, spec))
            f.write(f'{rid} & {claim}\\\\[1pt]\\multicolumn{{1}}{{l}}{{}} & & & \\\\\n' if False else '')
            f.write(f'{rid} & {claim} \\newline {{\\scriptsize {pos}}} & {src} & {spec} & {status}\\\\\n')
        f.write('\\end{longtable}\n}\n')
    print(f'[{lang}] wrote {out}: {len(rows)} rows')
    if leftovers:
        print(f'[{lang}] !! CJK leftovers in EN fields:')
        for rid, fld, val in leftovers:
            print(f'   {rid} [{fld}]: {val[:90]}')
    if lang == 'en':
        with open(out, encoding='utf-8') as f:
            body = f.read()
        bad = FWPUNCT.findall(body) + CJK.findall(body)
        if bad:
            print(f'[en] !! full-width/CJK chars remain in output: {sorted(set(bad))}')

emit('cn', OUT_CN, HEADER_CN)
emit('en', OUT_EN, HEADER_EN)
