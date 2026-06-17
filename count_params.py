import torch
import torch.nn as nn
from model_transformer import get_transformer_model

# Create model with your configuration
model = get_transformer_model(in_channels=1, num_classes=8, embed_dim=48, 
                              depths=[2,2,6,2], num_heads=[3,6,12,24], 
                              window_size=(7,7,7), decoder_dim=256)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count encoder vs components
encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())
gat_params = sum(p.numel() for p in model.mesh_gat.parameters())

print('===== MODEL PARAMETER STATISTICS =====')
print(f'Total Parameters: {total_params:,}')
print(f'Trainable Parameters: {trainable_params:,}')
print(f'Encoder Parameters: {encoder_params:,}')
print(f'Decoder Parameters: {decoder_params:,}')
print(f'GAT Head Parameters: {gat_params:,}')
print(f'Model Size (fp32): {total_params * 4 / (1024**2):.1f} MB')
print(f'Model Size (fp16): {total_params * 2 / (1024**2):.1f} MB')
print(f'\n===== COMPUTATIONAL COMPARISON =====')
print(f'Encoder Percentage: {100*encoder_params/total_params:.1f}%')
print(f'Decoder Percentage: {100*decoder_params/total_params:.1f}%')
print(f'GAT Percentage: {100*gat_params/total_params:.1f}%')
