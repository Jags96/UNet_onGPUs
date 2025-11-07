"""
Model factory for creating different architectures
"""
from models.unet import build_unet
from models.unet_plus_plus import build_unetpp


def create_model(config):
    """
    Factory function to create model based on config
    
    Args:
        config (dict): Model configuration
    
    Returns:
        nn.Module: Instantiated model
    """
    model_name = config['name'].lower()
    
    if model_name == 'unet':
        return build_unet(config)
    elif model_name in ['unet++', 'unetpp', 'unet_plus_plus']:
        return build_unetpp(config)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: unet, unet++")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get model information"""
    total_params = count_parameters(model)
    
    info = {
        'total_parameters': total_params,
        'total_parameters_M': total_params / 1e6,
        'model_size_MB': total_params * 4 / (1024 ** 2),  # float32
    }
    
    return info