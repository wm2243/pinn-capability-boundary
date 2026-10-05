# -*- coding: utf-8 -*-
"""verify_claims.py — 论文定量申明与仓库数据的一致性自动核对。

用法：python tools/verify_claims.py   （在 paper1_release 任意目录下运行均可）

检查项：
  A. claims_provenance.csv 结构完整（115 行、id 连续、字段数在已知错位集合内）
  B. CSV 中引用的数据文件/脚本在仓库中存在
  C. EXPERIMENT_REGISTRY.md 中引用的脚本与数据目录存在
  D. 两版 tex 无残留实验代号（白名单：C2 数学条件、ass:GC 标签、\texttt 内脚本名）
  E. 附录溯源表（provenance_cn/en.tex）与 CSV 同步（115 行、id 一致）
  F. 关键数值从源数据重算（C01/C02/C03/C04/C05/C14/C30/C62/C85/C99/C100/C115）
"""
import csv, json, re, sys, math
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # paper1_release/
PAPER = ROOT / 'paper'
DATA = ROOT / 'data'

fails = []
def check(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, ('| ' + detail if detail else ''))
    if not ok:
        fails.append(name)

def read_csv(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

# ---------- A. CSV 结构 ----------
csv_path = ROOT / 'claims_provenance.csv'
rows = list(csv.reader(open(csv_path, encoding='utf-8')))
hdr = rows[0]
body = [r for r in rows[1:] if r and r[0].strip()]
ids = [r[0] for r in body]
expect_ids = [f'C{i:02d}' for i in range(1, 116)]
check('A1 CSV 行数=115', len(body) == 115, f'actual {len(body)}')
check('A2 id 连续 C01..C115', ids == expect_ids)
# 已知错位：个别行因来源列含逗号少/多字段，合法集合含 9
bad_fields = [r[0] for r in body if len(r) not in (9, 10, 11, 12, 13)]
check('A3 字段数在已知错位集合内', not bad_fields, str(bad_fields))

D = {r[0]: r for r in body}  # 原始行（含错位，F 段不用 CSV 数值，只用文件路径）

# ---------- B. 数据文件/脚本存在性 ----------
SKIP_SRC = ('同上', '派生', '设计网格', '解析重算', 'Holm 重算', '——', '—')
missing = []
for r in body:
    src = r[5] if len(r) > 5 else ''
    if any(k in src for k in SKIP_SRC):
        continue
    for tok in re.split(r'[+、]| 与 ', src):
        tok = tok.strip().strip('（）()')
        m = re.search(r'([\w./\-一-鿿]+\.(?:csv|json|py))', tok)
        if not m:
            continue
        f = m.group(1)
        if f.endswith('.py'):
            # 脚本：在 code/ 下按文件名递归找
            hits = list((ROOT / 'code').rglob(Path(f).name))
            p = hits[0] if hits else ROOT / 'code' / '__nonexistent__' / f
        elif f.startswith('code/'):
            p = ROOT / f
        elif f.startswith('aggregated/') or f.startswith('raw/'):
            p = DATA / f
        elif '/' in f:
            p = DATA / 'aggregated' / f
        else:
            hits = list(DATA.rglob(f))
            p = hits[0] if hits else DATA / '__nonexistent__' / f
        if not p.exists():
            missing.append(f'{r[0]}: {f}')
check('B1 CSV 引用数据文件/脚本全部存在', not missing, '; '.join(missing[:8]))

# ---------- C. Registry 路径 ----------
# 反引号 token 只有两类：.py 脚本（code/ 下递归找）与 data/ 开头目录（支持 * 通配）
reg = (ROOT / 'EXPERIMENT_REGISTRY.md').read_text(encoding='utf-8')
miss_c = []
for tok in re.findall(r'`([^`]+)`', reg):
    tok = tok.strip()
    if tok.endswith('.py'):
        if not list((ROOT / 'code').rglob(tok)):
            miss_c.append(tok)
    elif tok.startswith('data/'):
        t = tok.rstrip('/')
        if '*' in t:
            if not list(ROOT.glob(t)):
                miss_c.append(tok)
        elif not (ROOT / t).exists():
            miss_c.append(tok)
    # 其余 token（函数名等）跳过
check('C1 Registry 脚本与数据目录存在', not miss_c, '; '.join(miss_c[:8]))

# ---------- D. tex 代号残留 ----------
CODES = ['E1b','Gstab-B','GstabB','E-new-3','Enew3','caseB','s8','S2a','S2c','S2d','S2e',
         'Gstab','G1b','G1c','B1b','B2b','F1d','P8','R1','R2','G2','G3','E1','E3',
         'C4','C5','D1','Vverify','S1','S2','S3','GC','C2']
C2_OK = re.compile(r'C1/C2|\(C2|C2\$|satisfies C2|满足 C2|需 C2|仅需 C1/C2')

def split_texttt(l):
    """按花括号配对剥离 \\texttt{...}，返回 (剥离后的行, 内容跨度列表)。排版修复会在
    texttt 内插入 \\hspace{0pt}（含嵌套花括号），朴素正则会截断导致漏剥。"""
    out, spans, i = [], [], 0
    while True:
        m = re.search(r'\\texttt\{', l[i:])
        if not m:
            out.append(l[i:])
            break
        s = i + m.start()
        out.append(l[i:s])
        j = s + len('\\texttt{')
        depth = 1
        while j < len(l) and depth:
            if l[j] == '{':
                depth += 1
            elif l[j] == '}':
                depth -= 1
            j += 1
        spans.append(l[s + len('\\texttt{'):j - 1])
        i = j
    return ''.join(out), spans

bad_hits = []
bad_tt = []
for f in ['paper1_v1_20.tex', 'paper1_v1_20_en.tex']:
    for i, l in enumerate((PAPER / f).read_text(encoding='utf-8').split('\n'), 1):
        if l.strip().startswith('%'):
            continue
        l2, tt_spans = split_texttt(l)
        l2 = re.sub(r'\\(label|ref)\{[^}]*\}', '', l2)
        for c in CODES:
            for m in re.finditer(r'(?<![A-Za-z0-9_\\])' + re.escape(c) + r'(?![A-Za-z0-9_])', l2):
                if c == 'C2' and C2_OK.search(l2):
                    continue
                if c == 'GC' and 'G-C' in l2:
                    continue
                bad_hits.append(f'{f}:{i}:{c}')
        # D2: \texttt 内不得混入论文编号（防批量改名污染脚本名）
        for sp in tt_spans:
            if re.search(r'实验|Experiment', sp):
                bad_tt.append(f'{f}:{i}:{sp}')
check('D1 tex 代号残留为零（白名单外）', not bad_hits, '; '.join(bad_hits[:8]))
check('D2 texttt 内无论文编号污染', not bad_tt, '; '.join(bad_tt[:8]))

# ---------- E. 溯源表同步 ----------
for f in ['provenance_cn.tex', 'provenance_en.tex']:
    t = (PAPER / f).read_text(encoding='utf-8')
    tids = re.findall(r'^(C\d+) & ', t, re.M)
    check(f'E1 {f} 115 行且 id 与 CSV 一致', tids == expect_ids, f'{len(tids)} rows')

# ---------- F. 数值重算 ----------
def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

# C04/C05 命中率
agg = read_csv(DATA / 'aggregated/v7_v1_exp_G1b_pentry/g1b_pentry_agg.csv')
pool = {r['family']: float(r['p_entry']) for r in agg if r['scope'] == 'pooled'}
check('F-C04 uniform 命中率 .6125', abs(pool.get('uniform', 0) - .6125) < 1e-9, str(pool.get('uniform')))
check('F-C05 三族命中率 .4625/.475/.3875',
      all(abs(pool[k] - v) < 1e-9 for k, v in [('rational', .4625), ('down_inv', .475), ('down_lin', .3875)]),
      str({k: pool.get(k) for k in ['rational', 'down_inv', 'down_lin']}))

# C01/C02/C03/C30 F1d
f1d = [r for r in read_csv(DATA / 'aggregated/v7_v2_exp_V2_F1d_real_jacobian/f1d_real_jacobian.csv')]
sub = [r for r in f1d if abs(float(r['nu']) - .005) < 1e-12 and int(r['N']) in (64, 128, 256, 512)]
imp = [1 - float(r['ratio_r']) for r in sub]
check('F-C01 改善范围 .126–.240', abs(min(imp) - .126) < .002 and abs(max(imp) - .240) < .002,
      f'{min(imp):.4f}–{max(imp):.4f}')
rr = [float(r['ratio_r']) for r in sub]
check('F-C02 ratio_r [.7596;.8741]', abs(min(rr) - .7596) < 1e-4 and abs(max(rr) - .8741) < 1e-4,
      f'{min(rr):.4f}/{max(rr):.4f}')
k1024 = [float(r['kappa_R']) for r in f1d if int(r['N']) == 1024]
check('F-C03 N=1024 kappa_R≈1.95e13', len(k1024) == 1 and abs(k1024[0] / 1.95e13 - 1) < 1e-3, str(k1024))
pr = [float(r['eff_rank_PR']) for r in f1d if abs(float(r['nu']) - .005) < 1e-12 and int(r['N']) in (256, 512, 1024, 2048)]
check('F-C30 eff_rank_PR≈109.1/109.4×3', len(pr) == 4 and abs(pr[0] - 109.10) < .05 and all(abs(x - 109.4) < .06 for x in pr[1:]),
      str([round(x, 2) for x in pr]))

# C14/C85 行数
n_e1b = sum(1 for _ in open(DATA / 'aggregated/v7_v1_exp_E1b_smooth/e1b_runs.csv', encoding='utf-8-sig')) - 1
check('F-C14 E1b 1584 runs', n_e1b == 1584, str(n_e1b))
n_poi = sum(1 for _ in open(DATA / 'aggregated/v7_v2_exp_V2_S2a_poisson_degenerate/poisson_runs.csv', encoding='utf-8-sig')) - 1
check('F-C85 Poisson 462 runs', n_poi == 462, str(n_poi))

# C62 E1 逐种子最优 β* 中位
e1 = read_csv(DATA / 'aggregated/v7_v1_exp_E1_beta_nu/e1_runs.csv')
meds = []
for nu in (.001, .003, .005, .01):
    seeds = {r['seed'] for r in e1 if abs(float(r['nu']) - nu) < 1e-12}
    best = []
    for s in seeds:
        rs = [r for r in e1 if abs(float(r['nu']) - nu) < 1e-12 and r['seed'] == s]
        best.append(float(min(rs, key=lambda r: float(r['L2']))['beta']))
    meds.append(med(best))
check('F-C62 逐种子最优 β* 中位 1/5/20/10', meds == [1.0, 5.0, 20.0, 10.0], str(meds))

# C99/C100 D1
d1 = read_csv(DATA / 'aggregated/v8_time_varying_d1r/d1r_runs.csv')
bmed = {}
bgood = {}
for b in (1.0, 3.0, 8.0, 15.0):
    v = [float(r['L2_total']) for r in d1 if float(r['beta']) == b]
    bmed[b] = med(v)
    bgood[b] = sum(1 for x in v if x < .06)
exp_med = {1.0: 1.66e-2, 3.0: 1.53e-2, 8.0: 5.56e-2, 15.0: 5.35e-2}
check('F-C99 D1 四档中位与好盆地数',
      all(abs(bmed[b] / exp_med[b] - 1) < .02 for b in exp_med) and [bgood[b] for b in (1.0, 3.0, 8.0, 15.0)] == [8, 9, 6, 8],
      f'med {[round(bmed[b], 4) for b in (1.0, 3.0, 8.0, 15.0)]} good {[bgood[b] for b in (1.0, 3.0, 8.0, 15.0)]}')

def wilcoxon_p_exact(x, y):
    """Wilcoxon 符号秩精确双侧 p（枚举 2^n 种符号分配；秩放大 2 倍避免半秩浮点）。"""
    d = [(abs(a - b), 1 if a > b else -1) for a, b in zip(x, y) if a != b]
    d.sort()
    n = len(d)
    r2, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and d[j + 1][0] == d[i][0]:
            j += 1
        avg2 = i + j + 2  # 2 * ((i+j)/2 + 1)
        r2.extend([avg2] * (j - i + 1))
        i = j + 1
    W2 = sum(rr for rr, (_, s) in zip(r2, d) if s > 0)
    dist = Counter()
    for signs in product((0, 1), repeat=n):
        dist[sum(rr * sg for rr, sg in zip(r2, signs))] += 1
    tot = 2 ** n
    lo = sum(c for k, c in dist.items() if k <= W2)
    hi = sum(c for k, c in dist.items() if k >= W2)
    return min(2 * min(lo, hi) / tot, 1.0)

base = {r['seed']: float(r['L2_total']) for r in d1 if float(r['beta']) == 1.0}
ps = []
for b in (3.0, 8.0, 15.0):
    oth = {r['seed']: float(r['L2_total']) for r in d1 if float(r['beta']) == b}
    ks = sorted(set(base) & set(oth))
    ps.append(wilcoxon_p_exact([oth[k] for k in ks], [base[k] for k in ks]))
check('F-C100 配对 Wilcoxon p≈.41/.32/.37',
      all(abs(p - e) < .03 for p, e in zip(ps, (.41, .32, .37))), str([round(p, 3) for p in ps]))

# C115 算术
check('F-C115 格数算术 297/1584/80',
      3 * 3 * 3 * 11 == 297 and 3 * 8 * 6 * 11 == 1584 and 4 * 20 == 80)

print()
if fails:
    print(f'FAILED: {len(fails)} 项 ->', '; '.join(fails))
    sys.exit(1)
print('ALL CHECKS PASSED')
