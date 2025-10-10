"""
RoboticsVLAModel definition for testing.

This file must be importable when loading the saved model.
"""

import torch
import torch.nn as nn


class RoboticsVLAModel(nn.Module):
    """Vision-Language-Action model for robotics tasks.
    
    Simplified to accept a single concatenated input for compatibility
    with the analysis agent's profiling system.
    """
    
    def __init__(self, input_dim=1280, hidden_dim=256, action_dim=7):
        super().__init__()
        
        # Input projection (simulates vision+language fusion)
        # 1280 = 512 (vision) + 768 (language)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Feature encoder (simulating ViT + BERT fusion)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with single concatenated input.
        
        Args:
            x: Tensor of shape (batch_size, input_dim) where input_dim = vision_dim + language_dim
        
        Returns:
            actions: Tensor of shape (batch_size, action_dim)
        """
        # Project input
        features = self.input_projection(x)
        
        # Encode features
        encoded = self.encoder(features)
        
        # Apply self-attention
        attended, _ = self.cross_attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        
        # Decode to actions
        actions = self.action_decoder(attended.squeeze(1))
        
        return actions
