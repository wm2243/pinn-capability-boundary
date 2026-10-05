# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:51:19 2026

@author: wwm
"""

import matplotlib.pyplot as plt
import numpy as np
import os

class PINNVisualizer:
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        # 🔥 核心修复：设置中文字体并解决负号显示问题
        # 优先使用微软雅黑，如果没有则回退到黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        # 强制让 matplotlib 使用普通的 ASCII 减号 (-) 而不是数学减号 (U+2212)
        plt.rcParams['axes.unicode_minus'] = False 
        
    def plot_1d_profile(self, x, u_pred, u_true, title="PINN vs Ground Truth"):
        """基础曲线：预测值 vs 真实值"""
        plt.figure(figsize=(8, 5))
        plt.plot(x, u_true, 'r--', label='True', linewidth=2)
        plt.plot(x, u_pred, 'b-', label='PINN', linewidth=2)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.xlabel('x'); plt.ylabel('u')
        plt.title(title)
        plt.savefig(f"{self.save_dir}/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()

    def plot_error_distribution(self, x, u_pred, u_true, title="Absolute Error Distribution"):
        """🔥 新增：误差分布图（双Y轴）
        上面是误差曲线，下面是绝对误差的柱状图，能一眼看出激波处的误差集中情况。
        """
        abs_err = np.abs(u_pred - u_true)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        # 上图：预测 vs 真实
        ax1.plot(x, u_true, 'r--', label='True', linewidth=2)
        ax1.plot(x, u_pred, 'b-', label='PINN', linewidth=2)
        ax1.set_ylabel('u'); ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)
        ax1.set_title(title)
        
        # 下图：绝对误差
        ax2.bar(x, abs_err, width=x[1]-x[0], color='orange', alpha=0.7)
        ax2.set_xlabel('x'); ax2.set_ylabel('Abs Error')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()

    def plot_training_curves(self, loss_history, l2_history, title="Training Convergence"):
        """训练收敛曲线（Loss + L2 Error）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.semilogy(loss_history); ax1.set_title('Total Loss'); ax1.grid(True, alpha=0.3)
        epochs = range(0, len(l2_history) * 100, 100)
        ax2.semilogy(epochs[:len(l2_history)], l2_history); ax2.set_title('L2 Error'); ax2.grid(True, alpha=0.3)
        plt.suptitle(title)
        plt.savefig(f"{self.save_dir}/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()
        
    def plot_sweep_results(self, csv_path, title="Beta Sweep Results"):
        """🔥 新增：参数扫描结果对比图
        自动读取 sweep_beta.py 生成的 CSV，画出 Beta vs L2 Error 的折线图。
        """
        import csv
        betas, l2_errors = [], []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                betas.append(float(row[0]))
                l2_errors.append(float(row[1]))
                
        plt.figure(figsize=(8, 5))
        plt.plot(betas, l2_errors, 'bo-', linewidth=2, markersize=8)
        plt.xscale('log')  # Beta 通常跨度大，用对数坐标
        plt.yscale('log')
        plt.xlabel('Beta (Weight Multiplier)'); plt.ylabel('L2 Error')
        plt.title(title); plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.save_dir}/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()