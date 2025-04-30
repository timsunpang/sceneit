import torch
import torch.nn as nn

class MetadataBackbone(nn.Module):
    def __init__(self, metadata_dim, output_size):
        super().__init__()
        self.metadata_proj = nn.Linear(metadata_dim, output_size)

    def forward(self, metadata):
        return self.metadata_proj(metadata.squeeze())
