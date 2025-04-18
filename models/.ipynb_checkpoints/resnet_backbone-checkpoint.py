import torch
import torch.nn as nn
from torchvision import models
import dataloader.get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ResNetBackBone(nn.Module):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

    def forward(self, x):
        with torch.no_grad():
            x = x.to(device)
            representations = self.feature_extractor(x).flatten(1)
        return representations

    def run_dataloader(self, dataloader):
        with torch.no_grad():
            for images, _ in dataloader:
                representations = self(images).cpu().numpy()
                yield representations

if __name__ == "__main__":
    resnet_backbone = ResNetBackBone().to("cuda")
    dataloader = get_dataloader("../clean_data/downloaded_posters", batch_size=1, shuffle=False)
    results = {"ResNetLatentVectors": [representations for representations in resnet_backbone.run_dataloader(dataloader)]}
    print(results)
