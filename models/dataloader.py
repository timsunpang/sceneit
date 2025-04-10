import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(image_directory):

    # Define the transformations to apply to each image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create an ImageFolder dataset
    dataset = datasets.ImageFolder(
        root=image_directory,
        transform=transform
    )

    # Create a DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=32,  # Define your batch size
        shuffle=True    # Shuffle the data
    )

    return data_loader