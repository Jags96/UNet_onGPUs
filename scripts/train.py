"""
Training script for U-Net fine-tuning with multi-GPU support
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.model_factory import create_model
from data.data_loader import get_dataloader
from utils.losses import get_loss_function
from utils.metrics import MetricTracker
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training"""
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Set up environment variables for torch.distributed
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # Single GPU or local multi-GPU
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = local_rank
        world_size = torch.cuda.device_count()
        
        if world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, logger, is_distributed):
    """Train for one epoch"""
    model.train()
    metric_tracker = MetricTracker()
    
    if is_distributed:
        dataloader.sampler.set_epoch(epoch)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}') if dist.get_rank() == 0 or not is_distributed else dataloader
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        metric_tracker.update(outputs, masks, loss.item())
        
        if batch_idx % 10 == 0 and (not is_distributed or dist.get_rank() == 0):
            metrics = metric_tracker.get_metrics()
            if isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'dice': f"{metrics['dice']:.4f}",
                    'iou': f"{metrics['iou']:.4f}"
                })
    
    return metric_tracker.get_metrics()


def validate(model, dataloader, criterion, device, logger):
    """Validate model"""
    model.eval()
    metric_tracker = MetricTracker()
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            metric_tracker.update(outputs, masks, loss.item())
    
    return metric_tracker.get_metrics()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training
    is_distributed = args.distributed and torch.cuda.device_count() > 1
    if is_distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = rank == 0
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
    
    # Setup logger
    logger = setup_logger(
        config['experiment']['name'], 
        config['experiment'].get('log_dir', 'logs'),
        rank=rank
    ) if is_main_process else None
    
    if is_main_process:
        logger.info(f"Starting training with config: {args.config}")
        logger.info(f"Device: {device}, Distributed: {is_distributed}")
    
    # Create model
    model = create_model(config['model'])
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create dataloaders
    train_loader = get_dataloader(
        config['data'],
        split='train',
        is_distributed=is_distributed
    )
    
    val_loader = get_dataloader(
        config['data'],
        split='val',
        is_distributed=False  # Validation on single GPU
    )
    
    # Loss and optimizer
    criterion = get_loss_function(config['training']['loss'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer, scheduler)
        if is_main_process:
            logger.info(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        if is_main_process:
            logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            scaler, device, epoch, logger, is_distributed
        )
        
        if is_main_process:
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Dice: {train_metrics['dice']:.4f}, "
                       f"IoU: {train_metrics['iou']:.4f}")
        
        # Validate
        if is_main_process:
            val_metrics = validate(model, val_loader, criterion, device, logger)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Dice: {val_metrics['dice']:.4f}, "
                       f"IoU: {val_metrics['iou']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['dice'] > best_dice
            best_dice = max(val_metrics['dice'], best_dice)
            
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_dice': best_dice,
                    'config': config
                },
                is_best,
                output_dir=args.output_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        scheduler.step()
        
        # Synchronize processes
        if is_distributed:
            dist.barrier()
    
    if is_distributed:
        dist.destroy_process_group()
    
    if is_main_process:
        logger.info(f"\nTraining completed! Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()