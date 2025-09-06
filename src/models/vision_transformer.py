"""
Vision Transformer (ViT) model implementation for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import contextlib
from typing import Optional


class VisionTransformer(nn.Module):
    """
    Vision Transformer implementation based on DINO architecture.
    """
    
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3,
                 num_classes: int = 100, embed_dim: int = 384, depth: int = 12,
                 num_heads: int = 6, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Size of image patches
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output logits of shape (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Extract class token
        cls_token_final = x[:, 0]
        
        # Classification head
        logits = self.head(cls_token_final)
        
        return logits
    
    def freeze(self, freeze_level: int):
        """
        Freeze model parameters based on level.
        
        Args:
            freeze_level: 0 = no freezing, 1 = freeze backbone, 2 = freeze all
        """
        if freeze_level == 0:
            # No freezing
            for param in self.parameters():
                param.requires_grad = True
        elif freeze_level == 1:
            # Freeze backbone (transformer blocks)
            for param in self.parameters():
                param.requires_grad = True
            for block in self.blocks:
                for param in block.parameters():
                    param.requires_grad = False
        elif freeze_level == 2:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
    
    def to_cuda(self):
        """Move model to CUDA device."""
        self.cuda()
    
    def debug(self):
        """Print model structure for debugging."""
        print("Vision Transformer Model Structure:")
        print(f"Image size: {self.img_size}")
        print(f"Patch size: {self.patch_size}")
        print(f"Number of patches: {self.num_patches}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Number of transformer blocks: {len(self.blocks)}")
        print(f"Number of attention heads: {self.blocks[0].num_heads}")
        print(f"MLP ratio: {self.blocks[0].mlp_ratio}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer."""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patched tensor of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size {H}x{W} must match {self.img_size}x{self.img_size}"
        
        # Apply convolution to get patches
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # Reshape to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # Multi-head self-attention
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias, attn_drop_rate, drop_rate
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop_rate: float = 0.0, proj_drop_rate: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention."""
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: int, drop_rate: float = 0.0):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def create_vision_transformer(model_size: str = "small", num_classes: int = 100) -> VisionTransformer:
    """
    Create a Vision Transformer with specified size.
    
    Args:
        model_size: Model size ("tiny", "small", "base", "large")
        num_classes: Number of output classes
        
    Returns:
        Configured Vision Transformer model
    """
    configs = {
        "tiny": {"embed_dim": 192, "depth": 6, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16}
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0
    )


@contextlib.contextmanager
def _isolate_hub_imports():
    """Avoid clash with project 'utils' when loading torch.hub DINO.

    - Temporarily remove project src paths from sys.path
    - Temporarily remove any loaded 'utils' modules from sys.modules
    """
    removed_paths = []
    removed_modules = {}
    for p in list(sys.path):
        if p.endswith('/src') or p.endswith('\\src'):
            removed_paths.append(p)
            sys.path.remove(p)
    for name in list(sys.modules.keys()):
        if name == 'utils' or name.startswith('utils.'):
            removed_modules[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        for p in removed_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
        sys.modules.update(removed_modules)


# === Hub-backed DINO backbone wrapper for head-only training (torch.hub, variable input size) ===
class DINOBackboneClassifier(nn.Module):
    """
    Wraps a DINO ViT-S/16 backbone from torch.hub and a trainable classifier head.

    - Backbone parameters are frozen (requires_grad=False) and set to eval mode
    - Forward runs backbone feature extraction under torch.no_grad()
    - Accepts variable input sizes (DINO interpolates positional embeddings)
    - Head remains trainable (384 -> num_classes)
    """

    def __init__(self, num_classes: int = 100, freeze_backbone: bool = True):
        super().__init__()
        with _isolate_hub_imports():
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # DINO ViT-S/16 embed dim = 384
        self.head = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # DINO models return features directly when called
            feats = self.backbone(x)
            # Handle different output shapes from DINO model
            if feats.dim() == 3:  # Shape: (batch_size, num_tokens, embed_dim)
                feats = feats[:, 0]  # Extract CLS token features
            elif feats.dim() == 2:  # Shape: (batch_size, embed_dim) - already extracted
                pass  # Use as is
            else:
                # Fallback: take mean across spatial dimensions if needed
                feats = feats.mean(dim=1) if feats.dim() > 2 else feats
        return self.head(feats)


# === LinearFlexibleDino: Flexible DINO model with configurable freezing ===
class LinearFlexibleDino(nn.Module):
    """
    Flexible DINO model that allows freezing/unfreezing backbone blocks.
    
    - Uses DINO ViT-S/16 backbone from torch.hub
    - Has a simple linear head (384 -> num_classes)
    - Supports freezing/unfreezing specific numbers of backbone blocks
    - Compatible with mask computation and sparse training
    """
    
    def __init__(self, num_classes: int = 100, num_layers_to_freeze: int = 12):
        """
        Initialize LinearFlexibleDino.
        
        Args:
            num_classes: Number of output classes
            num_layers_to_freeze: Number of backbone blocks to freeze (0-12)
        """
        super().__init__()
        
        # Load DINO backbone
        with _isolate_hub_imports():
            self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        
        # Simple linear head
        self.head = nn.Linear(384, num_classes)
        
        # Freeze specified number of layers
        self.freeze(num_layers_to_freeze)
        
    def freeze(self, num_layers_to_freeze: int = 12):
        """
        Freeze/unfreeze backbone blocks.
        
        Args:
            num_layers_to_freeze: Number of backbone blocks to freeze (0-12)
        """
        # Get all backbone blocks
        backbone_blocks = list(self.backbone.blocks.children())
        total_blocks = len(backbone_blocks)
        
        # Ensure num_layers_to_freeze is within valid range
        num_layers_to_freeze = max(0, min(num_layers_to_freeze, total_blocks))
        
        # Freeze/unfreeze blocks
        for i, block in enumerate(backbone_blocks):
            if i < num_layers_to_freeze:
                # Freeze this block
                for param in block.parameters():
                    param.requires_grad = False
                block.eval()
            else:
                # Unfreeze this block
                for param in block.parameters():
                    param.requires_grad = True
                block.train()
        
        # Always freeze embeddings and final norm
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        
        # pos_embed and cls_token are nn.Parameter objects, not modules
        self.backbone.pos_embed.requires_grad = False
        self.backbone.cls_token.requires_grad = False
        
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = False
        
        # Head is always trainable
        for param in self.head.parameters():
            param.requires_grad = True
            
        print(f"Frozen {num_layers_to_freeze}/{total_blocks} backbone blocks")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            feats = self.backbone(x)
            
            # Handle different output shapes from DINO model
            if feats.dim() == 3:  # Shape: (batch_size, num_tokens, embed_dim)
                feats = feats[:, 0]  # Extract CLS token features
            elif feats.dim() == 2:  # Shape: (batch_size, embed_dim) - already extracted
                pass  # Use as is
            else:
                # Fallback: take mean across spatial dimensions if needed
                feats = feats.mean(dim=1) if feats.dim() > 2 else feats
        
        # Apply linear head
        return self.head(feats)
    
    def get_trainable_parameters(self):
        """Get list of trainable parameter names for mask computation."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        return trainable_params
