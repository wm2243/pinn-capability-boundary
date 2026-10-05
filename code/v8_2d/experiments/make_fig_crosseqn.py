# -*- coding: utf-8 -*-
"""
第7章跨方程验证综合图：
(a) D1 时变 Burgers beta 扫描（L2 箱线图 + 好盆地率）
(b) 机制 A 直接检测（beta=1 vs beta=8 梯度指标）
(c) C4/C5 二维线性对流扩散 beta 扫描（L2 箱线图）
输出中英文双版 PDF。
"""
import json, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- 中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'cm'  # 数学符号用 Computer Modern，避免缺字形

_HERE = Path(__file__).resolve()
_REL = _HERE.parents[3]                    # paper1_release/
D1_ROOT = str(_REL/'data'/'raw'/'v8'/'time_varying'/'D1_timevarying_burgers_pytorch')
C5_ROOT = str(_REL/'data'/'raw'/'v8'/'burgers_2d'/'C5_advdiff2d_beta_scan')
OUT_DIR = os.environ.get('REPRO_OUT', str(_REL/'figures'/'repro'))
os.makedirs(OUT_DIR, exist_ok=True)

C_ORANGE = '#E8A24B'   # 增权族
C_GRAY   = '#9A9A9A'   # 等权
C_BLUE   = '#4B86B4'
C_RED    = '#C0392B'
C_GREEN  = '#3A8A18'


def load_beta(root, betas):
    data = {}
    for b in betas:
        files = sorted(glob.glob(os.path.join(root, f'beta_{b}', 'result_seed*.json')))
        l2, good = [], []
        for f in files:
            d = json.load(open(f, encoding='utf-8'))
            l2.append(d['L2_total'])
            good.append(bool(d.get('good_basin', False)))
        data[b] = {'l2': np.array(l2), 'good': np.array(good)}
    return data


def load_mechA(root, beta):
    files = sorted(glob.glob(os.path.join(root, f'beta_{beta}', 'result_seed*.json')))
    rows = []
    for f in files:
        d = json.load(open(f, encoding='utf-8'))
        m = d.get('mechanism_A')
        if m:
            rows.append(m)
    return rows


def make_fig(lang='zh'):
    """lang: 'zh' or 'en'"""
    is_zh = (lang == 'zh')
    L = lambda z, e: z if is_zh else e

    d1_betas = [1.0, 3.0, 8.0, 15.0]
    c5_betas = [1.0, 3.0, 8.0]
    d1 = load_beta(D1_ROOT, d1_betas)
    c5 = load_beta(C5_ROOT, c5_betas)
    ma1 = load_mechA(D1_ROOT, 1.0)
    ma8 = load_mechA(D1_ROOT, 8.0)

    def med(rows, key):
        return np.median([r[key] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={'width_ratios': [1.15, 1.0, 1.0]})

    # ---------- (a) D1 beta scan ----------
    ax = axes[0]
    positions = np.arange(len(d1_betas))
    box_data = [d1[b]['l2'] for b in d1_betas]
    bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(C_GRAY if d1_betas[i] == 1.0 else C_ORANGE)
        patch.set_alpha(0.75)
    # 散点
    for i, b in enumerate(d1_betas):
        x = np.random.normal(i, 0.05, size=len(d1[b]['l2']))
        ax.scatter(x, d1[b]['l2'], color='black', s=12, zorder=3, alpha=0.7)
    # 好盆地率标注
    for i, b in enumerate(d1_betas):
        rate = d1[b]['good'].mean()
        ax.text(i, ax.get_ylim()[1]*0.0 if i == 0 else 0, '', ha='center')
        ymin, ymax = 1e-3, 1.0
        ax.annotate(f'{int(rate*11)}/11', xy=(i, ymax*0.75), ha='center', fontsize=9,
                    color=C_GREEN if rate >= 0.8 else C_RED, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1.5e-3, 1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'$\\beta$={int(b) if b==int(b) else b}' for b in d1_betas])
    ax.set_ylabel(L('终态相对 $L_2$ 误差', 'Final relative $L_2$ error'))
    ax.set_title(L('(a) 时变 Burgers D1：$\\beta$ 扫描（$n=11$）',
                   '(a) Time-dependent Burgers D1: $\\beta$ scan ($n=11$)'), fontsize=10)
    ax.axhline(0.1, color=C_RED, ls='--', lw=0.8, alpha=0.6)
    ax.text(3.35, 0.11, L('好盆地阈值', 'basin thresh.'), fontsize=7.5, color=C_RED, ha='right')
    ax.grid(axis='y', alpha=0.25)

    # ---------- (b) Mechanism A ----------
    ax = axes[1]
    metrics = [
        ('grad_norm_ratio', L('梯度范数比\n$\\|g_W\\|/\\|g\\|$', 'Grad norm\nratio')),
        ('grad_cosine_similarity', L('方向余弦\n$\\cos(g_W,g)$', 'Direction\ncosine')),
        ('weighted_grad_row_norm_ratio', L('中心/边界\n有效梯度比', 'Center/boundary\neff. grad ratio')),
    ]
    x = np.arange(len(metrics))
    w = 0.35
    v1 = [med(ma1, k) for k, _ in metrics]
    v8 = [med(ma8, k) for k, _ in metrics]
    bars1 = ax.bar(x - w/2, v1, w, label=L('$\\beta=1$（等权）', '$\\beta=1$ (uniform)'),
                   color=C_GRAY, alpha=0.8)
    bars8 = ax.bar(x + w/2, v8, w, label=L('$\\beta=8$（中心增权）', '$\\beta=8$ (up-weight)'),
                   color=C_ORANGE, alpha=0.85)
    for bars, vals in [(bars1, v1), (bars8, v8)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.12, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], fontsize=9)
    ax.set_ylabel(L('数值（$n=11$ 中位）', 'Value (median of $n=11$)'))
    ax.set_title(L('(b) 机制 A 直接检测（冻结网络）', '(b) Direct test of Mechanism A (frozen net)'), fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 9.5)
    ax.grid(axis='y', alpha=0.25)

    # ---------- (c) C4/C5 beta scan ----------
    ax = axes[2]
    positions = np.arange(len(c5_betas))
    box_data = [c5[b]['l2'] for b in c5_betas]
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(C_GRAY if c5_betas[i] == 1.0 else C_ORANGE)
        patch.set_alpha(0.75)
    for i, b in enumerate(c5_betas):
        xx = np.random.normal(i, 0.05, size=len(c5[b]['l2']))
        ax.scatter(xx, c5[b]['l2'], color='black', s=12, zorder=3, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'$\\beta$={int(b) if b==int(b) else b}' for b in c5_betas])
    ax.set_ylabel(L('终态相对 $L_2$ 误差', 'Final relative $L_2$ error'))
    ax.set_title(L('(c) 2D 线性对流—扩散 C5（$n=11$）',
                   '(c) 2D linear advection-diffusion C5 ($n=11$)'), fontsize=10)
    ax.set_ylim(0.035, 0.075)
    # 标注 11/11
    for i in range(len(c5_betas)):
        ax.annotate('11/11', xy=(i, 0.072), ha='center', fontsize=9, color=C_GREEN, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_crosseqn_D1_C5_{lang}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('saved:', out)


if __name__ == '__main__':
    make_fig('zh')
    make_fig('en')
    print('done')
