"""
Data loading utilities for medical image segmentation datasets
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class SegmentationDataset(Dataset):
    """Generic segmentation dataset"""
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                  list(self.image_dir.glob('*.png')))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask


class KvasirDataset(Dataset):
    """Kvasir-SEG polyp segmentation dataset"""
    def __init__(self, root_dir, split='train', transform=None, image_size=(256, 256)):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Kvasir structure: images/ and masks/
        self.image_dir = self.root_dir / 'images'
        self.mask_dir = self.root_dir / 'masks'
        
        # Get all files and split
        all_files = sorted(list(self.image_dir.glob('*.jpg')))
        
        # 80-10-10 split
        train_split = int(0.8 * len(all_files))
        val_split = int(0.9 * len(all_files))
        
        if split == 'train':
            self.image_files = all_files[:train_split]
        elif split == 'val':
            self.image_files = all_files[train_split:val_split]
        else:  # test
            self.image_files = all_files[val_split:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask


class CVCDataset(Dataset):
    """CVC-ClinicDB polyp segmentation dataset"""
    def __init__(self, root_dir, transform=None, image_size=(256, 256)):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        # CVC structure: Original/ and Ground Truth/
        self.image_dir = self.root_dir / 'Original'
        self.mask_dir = self.root_dir / 'Ground Truth'
        
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask


def get_train_transforms(image_size=(256, 256)):
    """Training augmentations"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size=(256, 256)):
    """Validation transforms"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_dataloader(config, split='train', is_distributed=False):
    """Create dataloader based on config"""
    dataset_name = config['train_dataset'] if split == 'train' else config.get('test_dataset', config['train_dataset'])
    image_size = tuple(config.get('image_size', [256, 256]))
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)
    
    # Get transforms
    if split == 'train':
        transform = get_train_transforms(image_size)
    else:
        transform = get_val_transforms(image_size)
    
    # Create dataset
    if dataset_name == 'kvasir':
        dataset = KvasirDataset(
            root_dir=config['kvasir_path'],
            split=split,
            transform=transform,
            image_size=image_size
        )
    elif dataset_name == 'cvc':
        dataset = CVCDataset(
            root_dir=config['cvc_path'],
            transform=transform,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create sampler
    sampler = None
    shuffle = (split == 'train')
    
    if is_distributed and split == 'train':
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader