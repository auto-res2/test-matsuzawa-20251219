import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from omegaconf import DictConfig


class ResNet18(nn.Module):
    """ResNet-18 model for image classification."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Adjust final layer for num_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Post-init assertion: verify output layer configuration
        assert self.model.fc.out_features == num_classes, \
            f"Output layer size mismatch: expected {num_classes}, got {self.model.fc.out_features}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet50(nn.Module):
    """ResNet-50 model for image classification."""
    
    def __init__(self, num_classes: int = 100, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.model = models.resnet50(weights=weights)
        
        # Adjust final layer for num_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Post-init assertion: verify output layer configuration
        assert self.model.fc.out_features == num_classes, \
            f"Output layer size mismatch: expected {num_classes}, got {self.model.fc.out_features}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QuadraticModel(nn.Module):
    """Simple quadratic model for synthetic experiments: y = w^T x."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        
        # Post-init assertion: verify weight initialization
        assert self.weight.shape[0] == input_dim, \
            f"Weight dimension mismatch: expected {input_dim}, got {self.weight.shape[0]}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


def build_model(model_config: DictConfig) -> nn.Module:
    """Factory function to build models from config."""

    # Use 'architecture' field if available, otherwise fall back to 'name'
    # Remove hyphens and convert to lowercase for consistent matching
    if hasattr(model_config, 'architecture') and model_config.architecture:
        model_name = model_config.architecture.lower().replace("-", "")
    else:
        model_name = model_config.name.lower().replace("-", "")

    num_classes = model_config.get("num_classes", 10)
    pretrained = model_config.get("pretrained", False)

    if model_name == "resnet18":
        model = ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "quadratic":
        input_dim = model_config.get("input_dim", 1000)
        model = QuadraticModel(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Post-init assertion: verify model is properly initialized
    assert model is not None, "Model construction failed"
    assert hasattr(model, 'forward'), "Model missing forward method"
    
    return model
