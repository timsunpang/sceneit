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
from imdb import Cinemagoer
import pandas as pd
import sys
from urllib.error import HTTPError

def get_movie_by_id(links, id):
    id = links[links['movieId'] == id]['imdbId'].values[0]

    return id

if __name__ == "__main__":
    ratings = read_csv('../raw_data/ratings.csv')
    links = read_csv('../raw_data/links.csv')
    
    movies = {'imdbId': [], 'duration': [], 'imdb_score': [], 'genre_history': [], 'genre_thriller': [], 'genre_biography': [], 'genre_documentary': [], 'genre_animation': [], 'genre_war': [], 'genre_music': [], 'genre_family': [], 'genre_western': [], 'genre_fantasy': [], 'genre_drama': [], 'genre_musical': [], 'genre_adventure': [], 'genre_crime': [], 'genre_action': [], 'genre_film-noir': [], 'genre_comedy': [], 'genre_sport': [], 'genre_horror': [], 'genre_sci-fi': [], 'genre_mystery': [], 'genre_romance': []}

    movieIds = np.unique(ratings['movieId'])

    ia = IMDb()

    for movie_id in tqdm(movieIds, total = len(movieIds)):
        try:
            imdbId = get_movie_by_id(links, movie_id)
            movie = ia.get_movie(imdbId)
            genres = movie['genre'] 
            duration = movie['runtimes'][0]
            rating = movie['rating']

            movies['imdbId'].append(imdbId)
            movies['duration'].append(duration)
            movies['imdb_score'].append(rating)
            movies['genre_history'].append(int('History' in genres))
            movies['genre_thriller'].append(int('Thriller' in genres))
            movies['genre_biography'].append(int('Biography' in genres))
            movies['genre_documentary'].append(int('Documentary' in genres))
            movies['genre_animation'].append(int('Animation' in genres))
            movies['genre_war'].append(int('War' in genres))
            movies['genre_music'].append(int('Music' in genres))
            movies['genre_family'].append(int('Family' in genres))
            movies['genre_western'].append(int('Western' in genres))
            movies['genre_fantasy'].append(int('Fantasy' in genres))
            movies['genre_drama'].append(int('Drama' in genres))
            movies['genre_adventure'].append(int('Adventure' in genres))
            movies['genre_crime'].append(int('Crime' in genres))
            movies['genre_action'].append(int('Action' in genres))
            movies['genre_film-noir'].append(int('Film-Noir' in genres))
            movies['genre_comedy'].append(int('Comedy' in genres))
            movies['genre_sport'].append(int('Sport' in genres))
            movies['genre_horror'].append(int('Horror' in genres))
            movies['genre_sci-fi'].append(int('Sci-Fi' in genres))
            movies['genre_mystery'].append(int('Mystery' in genres))
            movies['genre_romance'].append(int('Romance' in genres))
            movies['genre_musical'].append(int('Musical' in genres))
        except:
            print('Error')

    for key in movies:
        print(key, len(movies[key]))

    pd.DataFrame(movies).to_csv('metadata_IMDB.csv', index=False)
