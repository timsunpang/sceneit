import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import re
from PIL import Image
from torch.utils.data import Dataset
from pandas import read_csv
from glob import glob
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
from imdb import IMDb

class PosterDataset(Dataset):
    def __init__(self, root_dir, transform, plots_path, links_path, movies_path,max_length = 512):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        print(f'Plots path: {plots_path}')
        print(f'Links path: {links_path}')
        print(f'Movies path: {movies_path}')
        print(f'Root dir: {root_dir}')
        print(f'Links path: {links_path}')
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(root_dir) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.plots_path = plots_path
        self.plots = read_csv(plots_path)
        self.links = read_csv(links_path)
        self.movies = read_csv(movies_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
        
    def __len__(self):
        return len(self.plots)
    
    def __getitem__(self, idx):
        #plot = self.plots[self.plots['Title'] == title][self.plots['Release Year'] == year]['Plot'].values[0]
        plot = self.plots.iloc[idx]['Plot']
        title = self.plots.iloc[idx]['Title']
        year = self.plots.iloc[idx]['Release Year']
        poster_path = glob(os.path.join(self.root_dir, self.plots.iloc[idx]['Title'] + '*.jpg'))
        if len(poster_path) == 0:
            image = np.zeros((3, 224, 224))
        else:
            image = Image.open(poster_path[0]).convert('RGB')
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
    
    def get_movie_by_id(self, id):
        id = self.links[self.links['movieId'] == id]['imdbId']
        ia = IMDb()
        plot = ia.get_movie(id).get('plot', [''])[0]
        if len(id) > 0:
            id = id.values[0]
            if id not in self.movies['imdbId'].values:
                print(f'Warning: imdbId {id} not found in movie posters')
                image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                image = self.transform(image)
                #plot = ""
                year = ""
                title = ""
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
                    'attention_mask': encoded['attention_mask'].squeeze(0)
                }
                return image, metadata
            title = self.movies[self.movies['imdbId'] == id]['Title'].values[0][:-7]

            year = self.plots[self.plots['Title'] == title]['Release Year']
            if len(year) == 0:
                year = ""
            else:
                year = year.values[0]

            poster_path = glob(os.path.join(self.root_dir, title + '*.jpg'))
            if len(poster_path) == 0:
                image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                image = Image.open(poster_path[0]).convert('RGB')
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
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
            return image, metadata
        else:
            print(f'Warning: No imdbPage found for movie {id}')
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            image = self.transform(image)
            year = ""
            title = ""
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
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
            return image, metadata

class User:
    def __init__(self, id, movie_dataset, ratings_path):
        self.id = id
        self.ratings = read_csv(ratings_path)
        self.ratings = self.ratings[self.ratings['userId'] == id]
        self.movies = []
        for movie_id, rating in tqdm(zip(self.ratings['movieId'], self.ratings['rating']), total = len(self.ratings['movieId'])):
            movie = movie_dataset.get_movie_by_id(movie_id)
            self.movies.append(movie)

    def sample_movies(self, profile_size = 10):
        found_movies = np.random.choice(self.movies, profile_size + 1, replace=False)
        return found_movies[:-1], found_movies[-1]

class UserDataset(Dataset):
    def __init__(self, ratings_path, poster_dataset, profile_size = 10):
        self.movielens_dataset = read_csv(ratings_path)
        self.users = []
        for id in np.unique(self.movielens_dataset['userId']):
            print(f'Create user {id}/{len(np.unique(self.movielens_dataset["userId"]))}', end = '\r')
            self.users.append(User(id, poster_dataset, ratings_path))
        self.profile_size = profile_size
    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx].sample_movies(self.profile_size)

def get_dataloader(image_directory, plots_path, movies_path, links_path, ratings_path,profile_size = 10, batch_size = 32, shuffle = True):

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
        plots_path=plots_path,
        movies_path=movies_path,
        links_path=links_path
    )
    user_dataset = UserDataset(ratings_path = ratings_path, poster_dataset = dataset, profile_size = profile_size)

    # Create a DataLoader
    data_loader = DataLoader(
        user_dataset,
        batch_size=batch_size,  # Define your batch size
        shuffle=shuffle    # Shuffle the data
    )

    return data_loader



if __name__ == "__main__":
    dataloader = get_dataloader("../clean_data/downloaded_posters/poster", "../raw_data/movie_plots.csv", "../clean_data/posters.csv", "../raw_data/links.csv", "../raw_data/ratings.csv", batch_size=1, shuffle=False)
    
    print(next(iter(dataloader)))