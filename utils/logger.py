"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_dir='logs', rank=0):
    """
    Setup logger for training
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to save logs
        rank (int): Process rank for distributed training
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Only setup handlers for main process
    if rank == 0:
        # Create log directory
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logger initialized. Logs will be saved to {log_file}")
    
    return logger


class TensorBoardLogger:
    """TensorBoard logging wrapper"""
    def __init__(self, log_dir='runs', experiment_name=None):
        from torch.utils.tensorboard import SummaryWriter
        
        if experiment_name:
            log_dir = Path(log_dir) / experiment_name
        
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
    
    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag, image, step):
        """Log image"""
        self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram"""
        self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close writer"""
        self.writer.close()


class WandbLogger:
    """Weights & Biases logging wrapper"""
    def __init__(self, project_name, experiment_name, config=None):
        try:
            import wandb
            self.wandb = wandb
            
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=config
            )
        except ImportError:
            print("wandb not installed. Skipping wandb logging.")
            self.wandb = None
    
    def log(self, metrics, step=None):
        """Log metrics"""
        if self.wandb:
            self.wandb.log(metrics, step=step)
    
    def log_image(self, key, image, step=None):
        """Log image"""
        if self.wandb:
            self.wandb.log({key: self.wandb.Image(image)}, step=step)
    
    def finish(self):
        """Finish logging"""
        if self.wandb:
            self.wandb.finish()