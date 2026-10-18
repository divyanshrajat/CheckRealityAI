"""
CheckReality AI - Intermediate Cross-Modal Attention Fusion Module (DR2)

STATUS: UNTRAINED, UNUSED ARCHITECTURAL SPECIFICATION.
DR2 Status: NOT MET.
Reason: A genuinely jointly labelled four-modality corpus (e.g. FakeAVCeleb / multimodal bench)
is not available in this workspace. CheckReality AI uses reliability-weighted late fusion
in production. This module implements the PyTorch Multi-Head Cross-Attention architecture
with key-padding masks for future research when jointly labelled corpora are acquired.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

CANONICAL_MODALITIES = ['video', 'audio', 'image', 'text']


class CrossModalAttentionFusion(nn.Module):
    """
    Intermediate Cross-Modal Attention Fusion Network:
    Projects heterogeneous unimodal feature representations into a shared latent space (embed_dim),
    applies multi-head self/cross-attention across modality tokens, and aggregates with masked pooling.
    Handles missing modalities via key-padding masks (True indicates missing / ignored position).
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear feature projection layers for heterogeneous model outputs
        self.proj_video = nn.Linear(1280, embed_dim)   # EfficientNet-B0 feature dim
        self.proj_audio = nn.Linear(512, embed_dim)    # ResNet-18 feature dim
        self.proj_image = nn.Linear(1536, embed_dim)   # EfficientNet-B3 feature dim
        self.proj_text = nn.Linear(768, embed_dim)     # RoBERTa-base hidden dim

        # Learned missing-modality token for absent streams
        self.missing_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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
        Forward pass with key-padding mask for missing modalities.
        modality_features: dict mapping modality name ('video', 'audio', 'image', 'text')
                           to [B, D_in] feature tensor.
        Returns:
            logits: [B, 2]
            attention_weights: [B, 4, 4] or None
        """
        device = next(self.parameters()).device
        # Determine batch size from first available feature tensor
        b = next(iter(modality_features.values())).size(0)

        tokens = []
        mask_list = []  # True = missing/padding (ignored by PyTorch MHA)

        # 1. Video
        if 'video' in modality_features:
            tokens.append(self.proj_video(modality_features['video']).unsqueeze(1))
            mask_list.append(torch.zeros((b, 1), dtype=torch.bool, device=device))
        else:
            tokens.append(self.missing_token.expand(b, 1, -1))
            mask_list.append(torch.ones((b, 1), dtype=torch.bool, device=device))

        # 2. Audio
        if 'audio' in modality_features:
            tokens.append(self.proj_audio(modality_features['audio']).unsqueeze(1))
            mask_list.append(torch.zeros((b, 1), dtype=torch.bool, device=device))
        else:
            tokens.append(self.missing_token.expand(b, 1, -1))
            mask_list.append(torch.ones((b, 1), dtype=torch.bool, device=device))

        # 3. Image
        if 'image' in modality_features:
            tokens.append(self.proj_image(modality_features['image']).unsqueeze(1))
            mask_list.append(torch.zeros((b, 1), dtype=torch.bool, device=device))
        else:
            tokens.append(self.missing_token.expand(b, 1, -1))
            mask_list.append(torch.ones((b, 1), dtype=torch.bool, device=device))

        # 4. Text
        if 'text' in modality_features:
            tokens.append(self.proj_text(modality_features['text']).unsqueeze(1))
            mask_list.append(torch.zeros((b, 1), dtype=torch.bool, device=device))
        else:
            tokens.append(self.missing_token.expand(b, 1, -1))
            mask_list.append(torch.ones((b, 1), dtype=torch.bool, device=device))

        # Stack into [B, 4, embed_dim]
        x = torch.cat(tokens, dim=1)
        # key_padding_mask: [B, 4]
        key_padding_mask = torch.cat(mask_list, dim=1)

        # Check that at least one modality is present
        if key_padding_mask.all():
            raise ValueError("At least one modality feature must be provided.")

        # Multi-Head Attention with key_padding_mask
        attn_out, attn_weights = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.layer_norm1(x + attn_out)

        # FFN block
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        # Masked mean pooling over only present tokens
        weights_float = (~key_padding_mask).float().unsqueeze(-1)  # [B, 4, 1]
        sum_pooled = torch.sum(x * weights_float, dim=1)           # [B, embed_dim]
        denom = torch.clamp(torch.sum(weights_float, dim=1), min=1.0)
        pooled = sum_pooled / denom

        logits = self.classifier(pooled)
        return logits, attn_weights
