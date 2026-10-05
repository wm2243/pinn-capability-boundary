# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:51:04 2026

@author: wwm
"""

import numpy as np

class PINNMetrics:
    """纯数学指标计算工具 (无状态)"""
    @staticmethod
    def compute_all(u_pred, u_true):
        l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        max_err = np.max(np.abs(u_pred - u_true))
        return {"L2_Error": l2, "Max_Error": max_err}