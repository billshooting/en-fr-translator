import matplotlib.pyplot as plt
import numpy as np
import os

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.learning_rates = []
        self.steps = []
        
        # 创建画布
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.ion()  # 开启交互模式
        
    def update(self, step, loss, lr):
        """ 更新训练数据 """
        self.steps.append(step)
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        
        # 清除旧图
        self.ax1.cla()
        self.ax2.cla()
        
        # 绘制Loss曲线
        self.ax1.plot(self.steps, self.train_losses, 'b-', linewidth=1)
        self.ax1.set_title('Training Loss')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        # 绘制Learning Rate曲线
        self.ax2.plot(self.steps, self.learning_rates, 'r-', linewidth=1)
        self.ax2.set_title('Learning Rate Schedule')
        self.ax2.set_xlabel('Training Steps')
        self.ax2.set_ylabel('Learning Rate')
        self.ax2.grid(True)
        
        # 自动调整布局
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # 短暂暂停更新图像
        
    def save_plots(self, save_dir="./plots"):
        """ 保存最终图表 """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存Loss曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.train_losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{save_dir}/loss_curve.png")
        plt.close()
        
        # 保存Learning Rate曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.learning_rates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(f"{save_dir}/lr_curve.png")
        plt.close()
