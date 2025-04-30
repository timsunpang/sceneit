import torch
import torch.nn as nn
from models.resnet_backbone import ResNetBackBone
from models.bert_backbone import BertBackbone
from models.metadata_backbone import MetadataBackbone

class MultimodalModel(nn.Module):
    def __init__(self, num_users, metadata_dim, embed_dim = 512, use_resnet = True, use_bert = True, use_metadata = True, device = 'cuda'):
        super().__init__()
        self.device = device

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        
        self.resnet_backbone = ResNetBackBone(embed_dim)
        self.bert_backbone = BertBackbone(embed_dim)
        self.metadata_backbone = MetadataBackbone(metadata_dim, output_size = embed_dim)
        num_used = 1
        if use_resnet:
            num_used += 1
        if use_bert:
            num_used += 1
        if use_metadata:
            num_used += 1

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_used, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1)  # Regression: normalized rating
        )
        self.use_resnet = use_resnet
        self.use_bert = use_bert
        self.use_metadata = use_metadata

    def forward(self, user_id, movie_id, input_ids, attention_mask, image, metadata):
        # User embedding
        user_emb = self.user_embedding(user_id)
        if self.use_resnet:
            img_emb = self.resnet_backbone(image)
        if self.use_bert:
            text_emb = self.bert_backbone(input_ids, attention_mask)
        if self.use_metadata:
            metadata_emb = self.metadata_backbone(metadata)
    
        # Combine all embeddings
        if self.use_resnet and self.use_bert and self.use_metadata:
            fused = torch.cat([user_emb, text_emb, img_emb, metadata_emb], dim=1)
        elif self.use_resnet and self.use_bert and not self.use_metadata:
            fused = torch.cat([user_emb, text_emb, img_emb], dim=1)
        elif self.use_resnet and not self.use_bert and self.use_metadata:
            fused = torch.cat([user_emb, img_emb, metadata_emb], dim=1)
        elif self.use_resnet and not self.use_bert and not self.use_metadata:
            fused = torch.cat([user_emb, img_emb], dim=1)
        elif not self.use_resnet and self.use_bert and self.use_metadata:
            fused = torch.cat([user_emb, text_emb, metadata_emb], dim=1)
        elif not self.use_resnet and self.use_bert and not self.use_metadata:
            fused = torch.cat([user_emb, text_emb], dim=1)
        elif not self.use_resnet and not self.use_bert and self.use_metadata:
            fused = torch.cat([user_emb, metadata_emb], dim=1)
        elif not self.use_resnet and not self.use_bert and not self.use_metadata:
            fused = torch.cat([user_emb], dim=1)
        return self.fusion(fused).squeeze()
