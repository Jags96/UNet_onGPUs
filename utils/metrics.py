"""
Evaluation metrics for segmentation
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Jaccard index)"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy


def precision_score(pred, target, smooth=1e-6):
    """Calculate precision"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_positive = (pred * target).sum(dim=(2, 3))
    predicted_positive = pred.sum(dim=(2, 3))
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.mean()


def recall_score(pred, target, smooth=1e-6):
    """Calculate recall (sensitivity)"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_positive = (pred * target).sum(dim=(2, 3))
    actual_positive = target.sum(dim=(2, 3))
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.mean()


def f1_score(pred, target):
    """Calculate F1 score"""
    prec = precision_score(pred, target)
    rec = recall_score(pred, target)
    
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    return f1


def specificity_score(pred, target, smooth=1e-6):
    """Calculate specificity"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_negative = ((1 - pred) * (1 - target)).sum(dim=(2, 3))
    actual_negative = (1 - target).sum(dim=(2, 3))
    
    specificity = (true_negative + smooth) / (actual_negative + smooth)
    return specificity.mean()


class MetricTracker:
    """Track multiple metrics during training/validation"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.losses = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.accuracies = []
    
    def update(self, pred, target, loss):
        """Update metrics with batch results"""
        with torch.no_grad():
            self.dice_scores.append(dice_coefficient(pred, target).item())
            self.iou_scores.append(iou_score(pred, target).item())
            self.losses.append(loss)
            self.precisions.append(precision_score(pred, target).item())
            self.recalls.append(recall_score(pred, target).item())
            self.f1_scores.append(f1_score(pred, target).item())
            self.accuracies.append(pixel_accuracy(pred, target).item())
    
    def get_metrics(self):
        """Get average metrics"""
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'loss': np.mean(self.losses),
            'precision': np.mean(self.precisions),
            'recall': np.mean(self.recalls),
            'f1': np.mean(self.f1_scores),
            'accuracy': np.mean(self.accuracies)
        }


def compute_confusion_matrix(pred, target):
    """Compute confusion matrix"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    return confusion_matrix(target_np, pred_np)


def hausdorff_distance(pred, target):
    """Compute Hausdorff distance (approximate)"""
    from scipy.spatial.distance import directed_hausdorff
    
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    distances = []
    for i in range(pred_np.shape[0]):
        pred_points = np.argwhere(pred_np[i, 0] > 0)
        target_points = np.argwhere(target_np[i, 0] > 0)
        
        if len(pred_points) > 0 and len(target_points) > 0:
            dist = max(
                directed_hausdorff(pred_points, target_points)[0],
                directed_hausdorff(target_points, pred_points)[0]
            )
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0