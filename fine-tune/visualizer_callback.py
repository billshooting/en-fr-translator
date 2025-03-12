from transformers import TrainerCallback
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from TrainingVisualizer import TrainingVisualizer

class VisualizationCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.visualizer = TrainingVisualizer()
        self.has_display = 'DISPLAY' in os.environ or os.name == 'nt'  # 检测图形界面支持

    def on_log(self, args, state, control, logs=None, **kwargs):
        """ 实时更新训练指标 """
        if not self.has_display:
            return  # 无图形界面时跳过显示
        
        # 仅处理包含训练损失的日志
        if logs and 'train_loss' in logs:
            # 获取当前训练步数
            current_step = state.global_step
            
            # 获取学习率（优先从日志读取）
            lr = logs.get('learning_rate')
            trainer = kwargs.get('trainer')
            
            # 如果日志没有学习率，从优化器获取
            if lr is None and trainer is not None:
                lr = trainer.optimizer.param_groups[0]['lr']
            
            # 更新可视化
            if lr is not None:
                self.visualizer.update(
                    step=current_step,
                    loss=logs['train_loss'],
                    lr=lr
                )

    def on_train_end(self, args, state, control, **kwargs):
        """ 训练结束时保存图表 """
        # 无论有无显示都保存
        self.visualizer.save_plots(args.output_dir)