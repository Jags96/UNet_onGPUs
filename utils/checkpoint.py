"""
Checkpoint saving and loading utilities
"""
import torch
import shutil
from pathlib import Path


def save_checkpoint(state, is_best, output_dir='checkpoints', filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this is the best model so far
        output_dir (str): Directory to save checkpoint
        filename (str): Checkpoint filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    torch.save(state, filepath)
    
    # Save as latest
    latest_path = output_dir / 'checkpoint_latest.pth'
    shutil.copyfile(filepath, latest_path)
    
    # If best, save separately
    if is_best:
        best_path = output_dir / 'checkpoint_best.pth'
        shutil.copyfile(filepath, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optimizer): Optimizer to load state
        scheduler (Scheduler): Scheduler to load state
    
    Returns:
        tuple: (start_epoch, best_metric)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Get epoch and metrics
    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_dice', 0.0)
    
    return start_epoch, best_metric


def resume_from_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint
        model (nn.Module): Model
        optimizer (Optimizer): Optimizer
        scheduler (Scheduler): Learning rate scheduler
        device (str): Device to load model to
    
    Returns:
        tuple: (model, optimizer, scheduler, start_epoch, best_metric)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    # Load optimizer
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_dice', 0.0)
    
    print(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    return model, optimizer, scheduler, start_epoch, best_metric