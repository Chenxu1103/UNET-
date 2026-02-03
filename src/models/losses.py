"""语义分割损失函数模块

实现 Dice Loss、CrossEntropy Loss 及其组合，用于模型训练
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiceLoss(nn.Module):
    """Dice 损失实现（针对多类 segmentation）

    Dice系数公式: Dice = 2|X∩Y| / (|X|+|Y|)

    优化：支持忽略背景、跳过GT中不存在的类别、类别加权
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        ignore_bg: bool = True,
        skip_empty: bool = True,
        class_weights: Optional[torch.Tensor] = None
    ) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg
        self.skip_empty = skip_empty
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算Dice损失
        
        Args:
            pred: 网络输出的logits张量 (N, C, H, W)
            target: 真实标签张量 (N, H, W)，值为类别ID
            
        Returns:
            Dice损失标量
        """
        # logits -> prob
        pred_prob = F.softmax(pred, dim=1)  # (N, C, H, W)

        # 将target转换为one-hot格式 (N, C, H, W)
        N, C, H, W = pred_prob.shape
        target_onehot = torch.zeros_like(pred_prob)
        # scatter_ 根据target索引填充one-hot
        target_onehot.scatter_(1, target.unsqueeze(1), 1)  # 在dim=1上填充

        # 计算每类的交集和并集
        pred_flat = pred_prob.view(N, C, -1)  # (N, C, H*W)
        target_flat = target_onehot.view(N, C, -1)  # (N, C, H*W)

        intersection = (pred_flat * target_flat).sum(dim=2)  # (N, C)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # (N, C)

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        # 选择要参与平均的类别：忽略背景/跳过空类
        valid = torch.ones((N, C), dtype=torch.bool, device=pred.device)
        if self.ignore_bg and C > 0:
            valid[:, 0] = False
        if self.skip_empty:
            # GT中该类像素为0 -> 不参与dice平均（小样本下非常关键）
            gt_sum = target_flat.sum(dim=2)  # (N,C)
            valid &= (gt_sum > 0)

        # 若这一batch除了背景全为空（极端情况），退化为对非背景直接平均
        if valid.sum() == 0:
            valid = torch.ones((N, C), dtype=torch.bool, device=pred.device)
            if self.ignore_bg and C > 0:
                valid[:, 0] = False

        # 加权平均（可选）
        if self.class_weights is not None:
            w = self.class_weights.to(pred.device).view(1, C).expand(N, C)
            w = torch.where(valid, w, torch.zeros_like(w))
            dice_mean = (dice_score * w).sum() / (w.sum() + 1e-6)
        else:
            dice_mean = dice_score[valid].mean()

        return 1.0 - dice_mean


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter (default=2.0). Higher gamma puts more focus on hard examples
        alpha: Weighting factor in [0, 1] for each class (default=None)
        ignore_index: Target value that is ignored and does not contribute to loss
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            pred: Network output logits (N, C, H, W)
            target: Ground truth labels (N, H, W)
        Returns:
            Focal loss scalar
        """
        # Get log probabilities
        log_prob = F.log_softmax(pred, dim=1)  # (N, C, H, W)

        # Get probabilities for the target class
        prob = torch.exp(log_prob)  # (N, C, H, W)

        # Gather the probabilities and log probs for the target class
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (N, C, H, W)
        prob_t = (prob * target_one_hot).sum(dim=1)  # (N, H, W)
        log_prob_t = (log_prob * target_one_hot).sum(dim=1)  # (N, H, W)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - prob_t) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target]  # (N, H, W)
            focal_weight = focal_weight * alpha_t

        # Compute focal loss
        loss = -focal_weight * log_prob_t

        # Mask out ignored indices
        mask = (target != self.ignore_index)
        loss = loss[mask].mean()

        return loss


class TverskyLoss(nn.Module):
    """Tversky Loss for imbalanced segmentation

    Generalization of Dice Loss with control over FP and FN:
    TL = 1 - (TP + smooth) / (TP + α*FN + β*FP + smooth)

    Args:
        alpha: Weight for false negatives (default=0.3). Higher alpha -> more penalty on FN -> higher recall
        beta: Weight for false positives (default=0.7). Higher beta -> more penalty on FP -> higher precision
        smooth: Smoothing constant (default=1e-5)
        ignore_bg: Whether to ignore background class
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        ignore_bg: bool = True
    ) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            pred: Network output logits (N, C, H, W)
            target: Ground truth labels (N, H, W)
        Returns:
            Tversky loss scalar
        """
        # Convert logits to probabilities
        pred_prob = F.softmax(pred, dim=1)  # (N, C, H, W)

        N, C, H, W = pred_prob.shape

        # Convert target to one-hot encoding
        target_onehot = torch.zeros_like(pred_prob)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)  # (N, C, H, W)

        # Flatten spatial dimensions
        pred_flat = pred_prob.view(N, C, -1)  # (N, C, H*W)
        target_flat = target_onehot.view(N, C, -1)  # (N, C, H*W)

        # Calculate True Positives, False Positives, False Negatives
        TP = (pred_flat * target_flat).sum(dim=2)  # (N, C)
        FP = (pred_flat * (1 - target_flat)).sum(dim=2)  # (N, C)
        FN = ((1 - pred_flat) * target_flat).sum(dim=2)  # (N, C)

        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)  # (N, C)

        # Select classes to average
        if self.ignore_bg:
            tversky = tversky[:, 1:]  # Ignore background class

        return 1.0 - tversky.mean()


class CombinedLoss(nn.Module):
    """组合损失：交叉熵 + Dice

    同时兼顾像素精度（CE）和区域准确性（Dice），
    特别对不平衡的小缺陷目标有帮助。

    优化：支持类别权重、Dice忽略背景、跳过空类
    """
    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        dice_ignore_bg: bool = True,
        dice_skip_empty: bool = True
    ) -> None:
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)  # 关键：类别权重
        self.dice = DiceLoss(ignore_bg=dice_ignore_bg, skip_empty=dice_skip_empty, class_weights=self.class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """计算组合损失

        Args:
            pred: 网络输出 logits (N, C, H, W)
            target: 真实标签 (N, H, W)

        Returns:
            tuple: (total_loss, ce_loss_value, dice_loss_value)
        """
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        return loss, ce_loss.item(), dice_loss.item()


class AdvancedCombinedLoss(nn.Module):
    """Advanced Combined Loss: Focal + Tversky + Dice

    Designed for high precision in cable/tape segmentation:
    - Focal Loss: Focuses on hard examples, reduces false positives
    - Tversky Loss: Balances precision vs recall with alpha/beta
    - Dice Loss: Ensures region overlap

    Args:
        weight_focal: Weight for focal loss (default=0.4)
        weight_tversky: Weight for tversky loss (default=0.4)
        weight_dice: Weight for dice loss (default=0.2)
        focal_gamma: Focusing parameter for focal loss
        tversky_alpha: FN weight for tversky (lower -> higher precision)
        tversky_beta: FP weight for tversky (higher -> higher precision)
        class_weights: Per-class weights
    """
    def __init__(
        self,
        weight_focal: float = 0.4,
        weight_tversky: float = 0.4,
        weight_dice: float = 0.2,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        class_weights: Optional[torch.Tensor] = None,
        dice_ignore_bg: bool = True
    ) -> None:
        super(AdvancedCombinedLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_tversky = weight_tversky
        self.weight_dice = weight_dice

        self.register_buffer("class_weights", class_weights)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, ignore_bg=dice_ignore_bg)
        self.dice = DiceLoss(ignore_bg=dice_ignore_bg, skip_empty=True, class_weights=class_weights)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float, float]:
        """计算组合损失

        Args:
            pred: 网络输出 logits (N, C, H, W)
            target: 真实标签 (N, H, W)

        Returns:
            tuple: (total_loss, focal_loss, tversky_loss, dice_loss)
        """
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        dice_loss = self.dice(pred, target)

        loss = (self.weight_focal * focal_loss +
                self.weight_tversky * tversky_loss +
                self.weight_dice * dice_loss)

        return loss, focal_loss.item(), tversky_loss.item(), dice_loss.item()
