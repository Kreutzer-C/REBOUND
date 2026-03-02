import math
import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR


class WarmupCosineScheduler(LambdaLR):
    """
    线性 Warmup + 余弦退火调度器。

    学习率变化规律：
        [0, warmup_epochs)    : 线性从 0 升至 base_lr
        [warmup_epochs, T_max]: 余弦退火从 base_lr 降至 eta_min
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min

        # 取第一个参数组的初始 lr 作为基准（构造时先记录，lambda 中使用）
        base_lrs = [pg['lr'] for pg in optimizer.param_groups]

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                # 线性 warmup：epoch=0 时 lr=0，epoch=warmup_epochs 时 lr=base_lr
                return float(epoch + 1) / float(warmup_epochs + 1)
            # 余弦退火阶段
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 缩放到 [eta_min/base_lr, 1.0]（取第一组 lr 作代表）
            min_ratio = eta_min / base_lrs[0] if base_lrs[0] > 0 else 0.0
            return min_ratio + (1.0 - min_ratio) * cosine

        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


# ──────────────────────────────────────────────────────────────────────────────
# 公开入口
# ──────────────────────────────────────────────────────────────────────────────

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    num_epochs: int,
    *,
    # step 调度器参数
    step_size: int   = 10,
    gamma: float     = 0.5,
    # cosine / cosine_warmup 共享参数
    eta_min: float   = 1e-6,
    # cosine_warmup 专用参数
    warmup_epochs: int = 5,
    last_epoch: int  = -1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    获取学习率调度器。

    Args:
        optimizer     : 已构建好的优化器。
        scheduler_name: 调度器类型，支持 'step' | 'cosine' | 'cosine_warmup'。
        num_epochs    : 总训练 epoch 数。
        step_size     : [step] 每隔多少 epoch 衰减一次学习率，默认 10。
        gamma         : [step] 衰减系数，默认 0.5。
        eta_min       : [cosine / cosine_warmup] 最小学习率，默认 1e-6。
        warmup_epochs : [cosine_warmup] warmup epoch 数，默认 5。
        last_epoch    : 上一个 epoch 编号（用于断点续训），默认 -1。

    Returns:
        对应的 lr_scheduler 实例。

    Example::

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = get(optimizer, 'cosine_warmup', num_epochs=100, warmup_epochs=5)
        for epoch in range(100):
            train(...)
            scheduler.step()
    """
    name = scheduler_name.lower()

    if name == 'step':
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )

    elif name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    elif name == 'cosine_warmup':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    else:
        raise ValueError(
            f"未知的调度器类型: '{scheduler_name}'，"
            f"请从 ['step', 'cosine', 'cosine_warmup'] 中选择。"
        )
