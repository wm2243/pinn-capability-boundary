# -*- coding: utf-8 -*-
"""六族权重纯函数自检（值域/恒等式/乘子界/势函数凸性/torch-np 一致性）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from losses.weight_families import (family_weight_np as fw, family_weight as fwt,
                                    FAMILIES, UP_FAMILIES, DOWN_FAMILIES,
                                    w_min_theory, stability_factor,
                                    multiplier_phi_rational, psi2pp_rational)

t = np.linspace(0, 50, 20001)
betas = [2.0, 5.0, 10.0]
allok = True
def chk(name, cond):
    global allok; allok &= bool(cond)
    print(("  [OK] " if cond else "  [FAIL] ") + name)

for b in betas:
    for fam in FAMILIES:
        w = fw(fam, t, b)
        if fam == "uniform":
            chk("%s b=%g 恒为1" % (fam, b), np.allclose(w, 1.0)); continue
        lo, hi = (1.0 / b, 1.0) if fam in DOWN_FAMILIES else (1.0, b)
        chk("%s b=%g 值域[%.3g,%.3g] 实测[%.4f,%.4f]" % (fam, b, lo, hi, w.min(), w.max()),
            w.min() >= lo - 1e-9 and w.max() <= hi + 1e-9)
    # rational 恒等式
    wr = fw("rational", t, b)
    chk("rational b=%g == (1+b t)/(1+t)" % b, np.allclose(wr, (1 + b * t) / (1 + t)))
    # 端点
    chk("rational b=%g t=0->1, t→∞->b" % b,
        abs(fw("rational", 0.0, b) - 1.0) < 1e-12 and abs(fw("rational", 1e12, b) - b) < 1e-6)
    chk("down_inv b=%g t=0->1, t→∞->1/b" % b,
        abs(fw("down_inv", 0.0, b) - 1.0) < 2e-12 and abs(fw("down_inv", 1e12, b) - 1.0 / b) < 1e-9)
    # 乘子界
    m = multiplier_phi_rational(t, b)
    chk("乘子 m_phi b=%g ∈ [2,2b]=[%.3g,%.3g] 实测[%.4f,%.4f]" % (b, 2, 2 * b, m.min(), m.max()),
        m.min() >= 2 - 1e-9 and m.max() <= 2 * b + 1e-9)
    # 势函数二阶导恒正
    chk("Psi2'' b=%g 恒正 (min=%.3e)" % (b, psi2pp_rational(t, b).min()),
        np.all(psi2pp_rational(t, b) > 0))
    # 稳定性常数
    chk("stability_factor b=%g: 增权=1, 减权=sqrt(b)=%.5f" % (b, np.sqrt(b)),
        all(abs(stability_factor(f, b) - 1.0) < 1e-12 for f in UP_FAMILIES)
        and all(abs(stability_factor(f, b) - np.sqrt(b)) < 1e-9 for f in DOWN_FAMILIES))

# torch / numpy 一致
try:
    import torch
    tt = torch.linspace(0, 20, 500)
    for fam in FAMILIES:
        for b in [3.0, 10.0]:
            a = fwt(fam, tt, b).numpy(); c = fw(fam, tt.numpy(), b)
            assert np.allclose(a, c, atol=1e-7), fam
    print("  [OK] torch 版与 numpy 版逐族逐点一致")
except ImportError:
    print("  [skip] 无 torch, 仅验 numpy")
print("=" * 60)
print("六族自检", "全部通过" if allok else "存在失败项")
sys.exit(0 if allok else 1)
