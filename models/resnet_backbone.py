import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ResNetBackBone(nn.Module):
    
    def __init__(self, output_size):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        self.image_proj = nn.Linear(2048, output_size)

    def forward(self, x):
        with torch.no_grad():
            x = x.to(device)
            representations = self.feature_extractor(x).flatten(1)
        return self.image_proj(representations)

if __name__ == "__main__":
    resnet_backbone = ResNetBackBone().to("cuda")
