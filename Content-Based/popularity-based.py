import argparse
import random
import numpy as np
import pandas as pd

from prepare_data import prepare_ml

from utils import get_ml_ratings_stats

import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset',dest='dataset',type=str,default=None,
        help="path to dataset folder to load")

    parser.add_argument(
        '--rec_type', dest='rec_type', type=str, default='damped_mean',
        help="Type of recommendation to be used, currently supports [damped_mean]")

    parser.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help="Seed for RNG reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    movies, ratings = prepare_ml(args.dataset)

    get_ml_ratings_stats(movies, ratings)

    movies = movies.sort_values(by='damped_mean_ratings', ascending=False).reset_index()

    for i in range(10):
        print(f'{i+1}\t{movies["title"][i]}\tvotes={movies["num_ratings"][i]}\tdmean={movies[f"{args.rec_type}_ratings"][i]}')
