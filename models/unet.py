"""
Standard U-Net implementation with pretrained encoder
"""
import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    """
    U-Net with pretrained ResNet encoder
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', 
                 in_channels=3, num_classes=1, activation=None):
        super(UNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    
    def forward(self, x):
        return self.model(x)


class CustomUNet(nn.Module):
    """
    Custom U-Net implementation from scratch (for reference)
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super(CustomUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.upconv4 = self.upconv(base_channels * 16, base_channels * 8)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)
        
        self.upconv3 = self.upconv(base_channels * 8, base_channels * 4)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = self.upconv(base_channels * 4, base_channels * 2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = self.upconv(base_channels * 2, base_channels)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


def build_unet(config):
    """Factory function to build U-Net model"""
    if config.get('custom', False):
        return CustomUNet(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            base_channels=config.get('base_channels', 64)
        )
    else:
        return UNet(
            encoder_name=config.get('encoder', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config['in_channels'],
            num_classes=config['num_classes']
        )