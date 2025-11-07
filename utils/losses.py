"""
Loss functions for segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class IoULoss(nn.Module):
    """IoU Loss (Jaccard Loss)"""
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss"""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        true_pos = (pred * target).sum(dim=(2, 3))
        false_neg = (target * (1 - pred)).sum(dim=(2, 3))
        false_pos = ((1 - target) * pred).sum(dim=(2, 3))
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        return 1 - tversky.mean()


class ComboLoss(nn.Module):
    """Combination of Dice and BCE loss"""
    def __init__(self, alpha=0.5, beta=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.alpha * dice_loss + self.beta * bce_loss


class DiceBCELoss(nn.Module):
    """Dice + BCE Loss (commonly used)"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for highly imbalanced data"""
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        true_pos = (pred * target).sum(dim=(2, 3))
        false_neg = (target * (1 - pred)).sum(dim=(2, 3))
        false_pos = ((1 - target) * pred).sum(dim=(2, 3))
        
        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        
        focal_tversky = (1 - tversky_index) ** self.gamma
        return focal_tversky.mean()


class BoundaryLoss(nn.Module):
    """Boundary loss for better edge detection"""
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_grad_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_grad_x = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_grad_y = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # Compute loss
        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2


def get_loss_function(loss_name):
    """Factory function to get loss by name"""
    losses = {
        'dice': DiceLoss(),
        'iou': IoULoss(),
        'focal': FocalLoss(),
        'tversky': TverskyLoss(),
        'dice_bce': DiceBCELoss(),
        'combo': ComboLoss(),
        'focal_tversky': FocalTverskyLoss(),
        'boundary': BoundaryLoss(),
        'bce': nn.BCEWithLogitsLoss()
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name]