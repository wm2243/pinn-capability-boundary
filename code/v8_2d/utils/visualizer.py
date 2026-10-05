# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:51:19 2026

@author: wwm
"""

import matplotlib.pyplot as plt

class PINNVisualizer:
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        # 🔥 核心修复：设置中文字体并解决负号显示问题
        # 优先使用微软雅黑，如果没有则回退到黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        # 强制让 matplotlib 使用普通的 ASCII 减号 (-) 而不是数学减号 (U+2212)
        plt.rcParams['axes.unicode_minus'] = False 
        
    def plot_1d_profile(self, x, u_pred, u_true, title="PINN vs Ground Truth"):
        plt.figure(figsize=(8, 5))
        plt.plot(x, u_true, 'r--', label='True', linewidth=2)
        plt.plot(x, u_pred, 'b-', label='PINN', linewidth=2)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.xlabel('x'); plt.ylabel('u')
        plt.title(title)
        plt.savefig(f"{self.save_dir}/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()

    def plot_training_curves(self, loss_history, l2_history, title="Training Convergence"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.semilogy(loss_history); ax1.set_title('Total Loss'); ax1.grid(True, alpha=0.3)
        epochs = range(0, len(l2_history) * 100, 100)
        ax2.semilogy(epochs[:len(l2_history)], l2_history); ax2.set_title('L2 Error'); ax2.grid(True, alpha=0.3)
        plt.suptitle(title)
        plt.savefig(f"{self.save_dir}/{title.replace(' ', '_')}.png", dpi=150)
        plt.close()