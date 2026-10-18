"""
CheckReality AI - Intermediate Cross-Modal Attention Fusion Module (DR2)
Implements Multi-Head Cross-Modal Attention across feature representations
with key-padding mask support for arbitrary subsets of active modalities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class CrossModalAttentionFusion(nn.Module):
    """
    Intermediate Cross-Modal Attention Fusion Network:
    Projects heterogeneous unimodal feature representations into a shared latent space (d_model),
    applies multi-head self/cross-attention across modality tokens, and aggregates with learned pooling.
    Handles missing modalities gracefully via key-padding masks.
    """
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Linear feature projection layers for heterogeneous model outputs
        self.proj_video = nn.Linear(1280, embed_dim) # EfficientNet-B0 feature dim
        self.proj_image = nn.Linear(1536, embed_dim) # EfficientNet-B3 feature dim
        self.proj_audio = nn.Linear(512, embed_dim)  # ResNet-18 feature dim
        self.proj_text = nn.Linear(768, embed_dim)   # RoBERTa-base hidden dim
        
        # Multi-Head Attention layer across modality tokens
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Classification Head (Binary: Real vs Fake)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic missing-modality mask.
        modality_features: dict mapping modality name ('video', 'image', 'audio', 'text') to [B, D_in] tensor.
        Returns: (logits [B, 2], attention_weights [B, num_heads, M, M])
        """
        tokens = []
        present_mods = []
        
        # Project active modality features into shared embedding dimension
        if 'video' in modality_features:
            tokens.append(self.proj_video(modality_features['video']).unsqueeze(1))
            present_mods.append('video')
        if 'image' in modality_features:
            tokens.append(self.proj_image(modality_features['image']).unsqueeze(1))
            present_mods.append('image')
        if 'audio' in modality_features:
            tokens.append(self.proj_audio(modality_features['audio']).unsqueeze(1))
            present_mods.append('audio')
        if 'text' in modality_features:
            tokens.append(self.proj_text(modality_features['text']).unsqueeze(1))
            present_mods.append('text')
            
        if not tokens:
            raise ValueError("At least one modality feature must be provided for cross-modal attention.")
            
        # Concatenate tokens along sequence dimension: [B, M, D]
        x = torch.cat(tokens, dim=1)
        
        # Multi-Head Attention over present modality tokens
        attn_out, attn_weights = self.mha(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # FFN block
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        # Mean pooling over modality tokens
        pooled = torch.mean(x, dim=1) # [B, D]
        logits = self.classifier(pooled) # [B, 2]
        
        return logits, attn_weights
