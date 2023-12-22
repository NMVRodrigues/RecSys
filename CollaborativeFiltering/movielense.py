import argparse
import random
import numpy as np
import pandas as pd

from prepare_data import prepare_ml

from utils import damped_mean

import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset',dest='dataset',type=str,default=None,
        help="path to dataset folder to load")

    parser.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help="Seed for RNG reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    movies, ratings = prepare_ml(args.dataset)

    damped_mean(ratings, 'rating', 'movieId', 10)

    movies['rating_damped_mean'] = movies['movieId'].map(ratings['rating_damped_mean'])
    movies['n_ratings'] = movies['movieId'].map(ratings.groupby("movieId")["rating"].count())
    movies = movies.sort_values(by='rating_damped_mean', ascending=False).reset_index()

    for i in range(100):
        print(f'{i}\t{movies["title"][i]}\tvotes={movies["n_ratings"][i]}\tdmean={movies["rating_damped_mean"][i]}')
