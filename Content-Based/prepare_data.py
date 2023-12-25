import pandas as pd
import os


def prepare_ml(folder):
    movies = pd.read_csv(os.path.join(folder, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(folder, 'ratings.csv'))

    return movies,ratings