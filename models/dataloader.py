import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import re
from PIL import Image
from torch.utils.data import Dataset
from pandas import read_csv
from glob import glob
from transformers import BertTokenizer
class PosterDataset(Dataset):
    def __init__(self, root_dir, transform, plots_path, max_length = 128):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(root_dir) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.plots_path = plots_path
        self.plots = read_csv(plots_path)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    def __len__(self):
        return len(self.plots)
    
    def __getitem__(self, idx):
        #plot = self.plots[self.plots['Title'] == title][self.plots['Release Year'] == year]['Plot'].values[0]
        plot = self.plots.iloc[idx]['Plot']
        title = self.plots.iloc[idx]['Title']
        year = self.plots.iloc[idx]['Release Year']
        image = Image.open(glob(os.path.join(self.root_dir, self.plots.iloc[idx]['Title'] + '*.jpg'))[0]).convert('RGB')
        image = self.transform(image)

        encoded = self.tokenizer(
            plot,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        metadata = {
            'title': title,
            'year': year,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'plot': plot  # Including original plot text just in case
        }
        return image, metadata

def get_dataloader(image_directory, plots_path,batch_size = 32, shuffle = True):

    # Define the transformations to apply to each image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create an ImageFolder dataset
    dataset = PosterDataset(
        root_dir=image_directory,
        transform=preprocess,
        plots_path=plots_path
    )

    # Create a DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Define your batch size
        shuffle=shuffle    # Shuffle the data
    )

    return data_loader

if __name__ == "__main__":
    dataloader = get_dataloader("../clean_data/downloaded_posters/poster", "../clean_data/plots.csv", batch_size=1, shuffle=False)
    print(next(iter(dataloader)))