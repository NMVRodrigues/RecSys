import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import random
import numpy as np

from sklearn.metrics.pairwise import linear_kernel

from utils.mlense_functions import prepare_ml, fix_ml, get_ml_ratings_stats
from utils.mlense_functions import find_closest_title, get_index_from_title, get_title_from_index
from utils.tfidf import tfidf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset',dest='dataset', type=str, required=True,
        help="path to dataset folder to load [only supports movie-lense for now]")

    parser.add_argument(
        '--movie', dest='movie', type=str, required=True,
        help="Movie from which we want recommendations")

    parser.add_argument(
        '--n_rec', dest='n_rec', type=int, default=10,
        help="Number of movies to recommend")

    parser.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help="Seed for RNG reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    movies, _ = prepare_ml(os.path.join('Datasets', args.dataset))
    movies = fix_ml(movies)

    docs = []
    document_words = []
    for ind in movies.index:
        docs.append(movies['movieId'][ind])
        document_words.append(movies['genres'][ind].split('|'))

    tfidf_matrix = tfidf(docs, document_words, True)

    # create the cosine similarity matrix
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    closest_title, distance_score = find_closest_title(movies, args.movie)
    # When a user does not make misspellings
    if distance_score == 100:
        movie_index = get_index_from_title(movies, closest_title)
        movie_list = list(enumerate(sim_matrix[int(movie_index)]))
        # remove the typed movie itself
        similar_movies = list(
            filter(lambda x: x[0] != int(movie_index), sorted(movie_list, key=lambda x: x[1], reverse=True)))

        print('Here\'s the list of movies similar to ' + '\033[1m' + str(closest_title) + '\033[0m' + '.\n')
        for i, s in similar_movies[:args.n_rec]:
            print(get_title_from_index(movies=movies, index=i))
    # When a user makes misspellings
    else:
        print('Did you mean ' + '\033[1m' + str(closest_title) + '\033[0m' + '?', '\n')
        movie_index = get_index_from_title(movies, closest_title)
        movie_list = list(enumerate(sim_matrix[int(movie_index)]))
        similar_movies = list(
            filter(lambda x: x[0] != int(movie_index), sorted(movie_list, key=lambda x: x[1], reverse=True)))
        print('Here\'s the list of movies similar to ' + '\033[1m' + str(closest_title) + '\033[0m' + '.\n')
        for i, s in similar_movies[:args.n_rec]:
            print(get_title_from_index(movies=movies, index=i))