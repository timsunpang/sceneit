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
import pandas as pd
import sys

def get_movie_by_id(links, id):
    id = links[links['movieId'] == id]['imdbId'].values[0]
    ia = IMDb()
    plot = ia.get_movie(id).get('plot', [''])[0]
    return id, plot

if __name__ == "__main__":
    ratings = read_csv('../raw_data/ratings.csv')
    links = read_csv('../raw_data/links.csv')
    
    movies = {'imdbId': [], 'plot': []}
    movieIds = np.unique(ratings['movieId']) 

    for movie_id in tqdm(movieIds, total = len(movieIds)):
        try:
            plot, imdbId = get_movie_by_id(links, movie_id)
            movies['plot'].append(plot)
            movies['imdbId'].append(imdbId)
        except:
            print('Error')
    pd.DataFrame(movies).to_csv('plots_Imdb.csv', index=False)