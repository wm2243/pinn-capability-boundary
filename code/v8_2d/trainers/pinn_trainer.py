# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:51:39 2026

@author: wwm
"""

import torch
from utils.metrics import PINNMetrics

class PINNTrainer:
    """
    PINN 训练器
    负责编排：采样 -> 物理残差计算 -> Loss 聚合 -> 反向传播 -> 监控
    """
    def __init__(self, model, pde_engine, criterion, optimizer, sampler, visualizer, device):
        self.model = model
        self.pde_engine = pde_engine      # 物理引擎
        self.criterion = criterion        # Loss 聚合器
        self.optimizer = optimizer
        self.sampler = sampler
        self.visualizer = visualizer
        self.device = device
        self.loss_history, self.l2_history = [], []

    def train(self, config, test_data, epochs=3000):
        self.model.train()
        for epoch in range(epochs):
            # 1. 采样
            data = self.sampler.sample(config['n_pde'], config['n_bc'])
            
            # 2. 计算纯粹的物理残差
            pde_res = self.pde_engine.compute_residual(self.model, data['x_pde'], data['t_pde'])
            
            # 3. 计算边界预测
            bc_pred = self.model(data['x_bc'], data['t_bc'])
            
            # 4. 初始条件预测 (瞬态问题)
            ic_pred, ic_true = None, None
            if 'x_ic' in data:
                ic_pred = self.model(data['x_ic'], data['t_ic'])
                ic_true = test_data.get('u_ic')
            
            # 5. Loss 聚合
            total_loss, loss_dict = self.criterion(pde_res, bc_pred, test_data['u_bc'], ic_pred, ic_true)
            
            # 6. 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 7. 监控
            self.loss_history.append(total_loss.item())
            if epoch % 100 == 0:
                metrics = self.evaluate(test_data)
                self.l2_history.append(metrics["L2_Error"])
                print(f"Epoch {epoch:04d} | Loss: {total_loss.item():.4e} | L2: {metrics['L2_Error']:.4e}")

    def evaluate(self, test_data):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(test_data['x'], test_data['t']).detach().cpu().numpy().flatten()
        return PINNMetrics.compute_all(u_pred, test_data['u_true'])