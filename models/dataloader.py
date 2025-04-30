import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import re
from PIL import Image
from torch.utils.data import Dataset
from pandas import read_csv
import pandas as pd
from glob import glob
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
from imdb import IMDb
import torch

class MovieDataset(Dataset):
    def __init__(self, plots_path, links_path, posters_root_dir, posters_path, metadata_path, ratings_path, transform, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        self.transform = transform
        
        plots = read_csv(plots_path)
        posters = read_csv(posters_path, encoding = 'latin-1')
        links = read_csv(links_path)
        metadata = read_csv(metadata_path)
        self.ratings = read_csv(ratings_path)

        self.image_dir = posters_root_dir

        self.df = pd.merge(plots, links, on = 'imdbId', how = 'outer')
        self.df = pd.merge(metadata, self.df, on = 'imdbId', how = 'outer')
        self.df = pd.merge(posters, self.df, on = 'imdbId', how = 'outer')

        self.metadata_cols = list(metadata.columns)[1:]
    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        userId = self.ratings.iloc[idx]['userId']
        rating = self.ratings.iloc[idx]['rating']
        movieId = self.ratings.iloc[idx]['movieId']
        row = self.df[self.df['movieId'] == movieId]
        # Text
        plot = row['plot'].values[0]
        if type(plot) == float:
            plot = "Plot is invalid"
        encoded = self.tokenizer(
            plot,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Image
        poster_path = glob(os.path.join(self.image_dir, f"*{row['Title'].values[0]}*.jpg"))
        if len(poster_path) > 0:
            image = Image.open(poster_path[0]).convert("RGB")
            image = self.transform(image)
        else:
            image = Image.fromarray(np.zeros((256, 256))).convert("RGB")
            image = self.transform(image)
        # Metadata
        metadata = torch.nan_to_num(torch.tensor(row[self.metadata_cols].values[0].astype(np.float32)))
        if metadata.shape[0] == 2:
            print(row)

        return {
            'user_id': torch.tensor(userId, dtype=torch.long),
            'movie_id': torch.tensor(movieId, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float32),  # or 'rating_norm'
            'plot': plot,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'image': image,
            'metadata': metadata
        }

def get_dataloader(plots_path, links_path, posters_root_dir, posters_path, metadata_path, ratings_path, batch_size = 32, shuffle = True):

    # Define the transformations to apply to each image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create an ImageFolder dataset
    dataset = MovieDataset(plots_path = plots_path, links_path = links_path, posters_root_dir = posters_root_dir, posters_path = posters_path, metadata_path = metadata_path, ratings_path = ratings_path, transform = preprocess)

    # Create a DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Define your batch size
        shuffle=shuffle    # Shuffle the data
    )

    return dataset, data_loader



if __name__ == "__main__":
    dataloader = get_dataloader(plots_path = '../clean_data/plots_Imdb_corrected.csv', links_path = '../raw_data/links.csv', posters_root_dir = '../clean_data/downloaded_posters/poster/', posters_path = '../raw_data/movie_posters.csv', metadata_path = '../clean_data/metadata_IMDB.csv', ratings_path = '../raw_data/ratings.csv', batch_size=1, shuffle=False)
    
    print(next(iter(dataloader)))
