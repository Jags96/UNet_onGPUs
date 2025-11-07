# UNet_onGPUs
# U-Net Fine-tuning for Multi-Dataset Segmentation

Professional repository for fine-tuning U-Net and U-Net++ on medical imaging datasets with multi-GPU SLURM support.

## Repository Structure

```
unet-finetuning/
├── README.md
├── requirements.txt
├── setup.py
├── environment.yml
├── .gitignore
├── configs/
│   ├── base_config.yaml
│   ├── unet_config.yaml
│   └── unetpp_config.yaml
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   ├── transforms.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── unet.py
│   ├── unet_plus_plus.py
│   └── model_factory.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── losses.py
│   ├── logger.py
│   └── checkpoint.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── download_data.sh
├── slurm/
│   └── train_multi_gpu.slurm
└── notebooks/
    └── data.ipynb

```

## Datasets

**Training Dataset (A):** Kvasir-SEG (Polyp Segmentation)
- 1000 polyp images with segmentation masks
- GI tract endoscopy images

**Testing Datasets:**
- **Dataset A:** Kvasir-SEG test split
- **Dataset B:** CVC-ClinicDB (Polyp Segmentation)
  - 612 polyp images with ground truth
  - Similar domain, different acquisition setup

Both datasets are publicly available and medically relevant for GI tract polyp detection.

## Features

- ✅ Standard U-Net with pretrained encoder (ResNet34/50)
- ✅ U-Net++ variant with dense skip connections
- ✅ Multi-GPU training with DistributedDataParallel
- ✅ SLURM integration for HPC clusters
- ✅ Comprehensive logging (TensorBoard, Weights & Biases)
- ✅ Mixed precision training (AMP)
- ✅ Multiple loss functions (Dice, BCE, Focal, Combo)
- ✅ Cross-dataset evaluation
- ✅ Model checkpointing and resuming
- ✅ Extensive metrics (IoU, Dice, Precision, Recall)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/unet-finetuning.git
cd unet-finetuning

# Create conda environment
conda env create -f environment.yml
conda activate unet-finetune

# Or use pip
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Download Datasets

```bash
bash scripts/download_data.sh
```

### 2. Local Training (Single GPU)

```bash
python scripts/train.py \
    --config configs/unet_config.yaml \
    --gpu 0
```

### 3. SLURM Multi-GPU Training

```bash
sbatch slurm/train_multi_gpu.slurm
```

### 4. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset kvasir \
    --config configs/unet_config.yaml
```

## Configuration

All configurations are in YAML format. Example `configs/unet_config.yaml`:

```yaml
model:
  name: "unet"
  encoder: "resnet34"
  pretrained: true
  in_channels: 3
  num_classes: 1

data:
  train_dataset: "kvasir"
  test_datasets: ["kvasir", "cvc"]
  batch_size: 16
  num_workers: 4
  image_size: [256, 256]

training:
  epochs: 100
  learning_rate: 1e-4
  optimizer: "adam"
  scheduler: "cosine"
  loss: "dice_bce"
  
distributed:
  backend: "nccl"
  world_size: 4
```

## SLURM Scripts

### Multi-GPU Training

```bash
#!/bin/bash
#SBATCH --job-name=unet-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

srun python scripts/train.py \
    --config configs/unet_config.yaml \
    --distributed
```

## Model Architectures

### Standard U-Net
- Encoder: Pretrained ResNet34 from torchvision
- Decoder: Symmetric upsampling path
- Skip connections: Concatenation

### U-Net++ (Modified Variant)
- Dense skip pathways between encoder-decoder
- Deep supervision
- Pruned version for efficiency

## Results

Results will be logged to:
- `checkpoints/` - Model weights
- `logs/` - Training logs
- `results/` - Evaluation metrics and visualizations
- TensorBoard: `tensorboard --logdir runs/`

## Citation

If you use this code, please cite:

```bibtex
@misc{unet-finetuning-2024,
  author = {Your Name},
  title = {U-Net Fine-tuning for Medical Image Segmentation},
  year = {2024},
  url = {https://github.com/yourusername/unet-finetuning}
}
```

## License

MIT License

## Acknowledgments

- Kvasir-SEG Dataset: [Paper](https://arxiv.org/abs/1911.07069)
- CVC-ClinicDB Dataset: [Paper](https://link.springer.com/article/10.1007/s00535-015-1084-3)
- Segmentation Models PyTorch: [GitHub](https://github.com/qubvel/segmentation_models.pytorch)
