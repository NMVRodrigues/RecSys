import sys
import os
import argparse
import random
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from sklearn.metrics.pairwise import linear_kernel

from utils.mlense_functions import prepare_ml, fix_ml, get_ml_ratings_stats
from utils.mlense_functions import find_closest_title, get_index_from_title, get_title_from_index
from utils.tfidf import tfidf
from utils._print_styles import Text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        '--dataset',dest='dataset', type=str,
        default='C://Users//Utilizador//Desktop//RecSys//Datasets//ml-small',
        help="path to dataset folder to load [only supports movie-lense for now]")

    parser.add_argument(
        '--movie', dest='movie', type=str, required=True,
        help="Movie from which we want recommendations")

    parser.add_argument(
        '--n_rec', dest='n_rec', type=int, default=10,
        help="Number of movies to recommend")

    parser.add_argument(
        '--ratings', dest='ratings', action="store_true",
        help="Whether to present a rating ranked version of the similar results")

    parser.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help="Seed for RNG reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    movies, ratings = prepare_ml(args.dataset)
    movies = fix_ml(movies)

    if args.ratings is True:
        get_ml_ratings_stats(movies, ratings)
        movies = movies.sort_values(by='damped_mean_ratings', ascending=False).reset_index()

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
    if distance_score != 100:
        print('Did you mean ' + Text.BOLD + closest_title + Text.END + '?', '\n')
    movie_index = get_index_from_title(movies, closest_title)
    movie_list = list(enumerate(sim_matrix[int(movie_index)]))
    # remove the typed movie itself
    similar_movies = list(
        filter(lambda x: x[0] != int(movie_index), sorted(movie_list, key=lambda x: x[1], reverse=True)))

    print('Here\'s the list of movies similar to ' + Text.BOLD + closest_title + Text.END + '.\n')
    similar_movies_titles = []
    for i, s in similar_movies[:args.n_rec]:
        similar_movie_title = get_title_from_index(movies=movies, index=i)
        similar_movies_titles.append(similar_movie_title)
        print(similar_movie_title)

    if args.ratings is True:
        print('\nHere\'s the list of movies similar to ' + Text.BOLD + closest_title + Text.END +
              ' ordered by popularity.\n')
        similar_movies_sorted = movies.loc[movies['title'].isin(similar_movies_titles)]
        similar_movies_sorted = similar_movies_sorted.sort_values(by='damped_mean_ratings', ascending=False).reset_index()
        for ind in similar_movies_sorted.index:
            print(f'{similar_movies_sorted["title"][ind]} - rating: {round(similar_movies_sorted["damped_mean_ratings"][ind], 2)}')
