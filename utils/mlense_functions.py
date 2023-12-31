import pandas
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('ignore')

from .utils import damped_mean, levenshtein_distance


def prepare_ml(folder: str) -> (pd.DataFrame, pd.DataFrame):
    movies = pd.read_csv(os.path.join(folder, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(folder, 'ratings.csv'))

    return movies, ratings


def fix_ml(movies:pd.DataFrame) -> pd.DataFrame:

    # remove all movies that do not have a release year
    movies = movies[movies['title'].str.contains(r'\(\d{4}\)', na=False)]

    # create new column for the years
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')

    # remove parentheses from title
    movies['title'] = movies['title'].apply(lambda title: title.split(' (')[0])

    movies = movies[~(movies['genres'] == '(no genres listed)')].reset_index(drop=True)

    # change 'Sci-Fi' to 'SciFi' and 'Film-Noir' to 'Noir'
    movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi')
    movies['genres'] = movies['genres'].str.replace('Film-Noir', 'Noir')

    return movies


def get_ml_ratings_stats(movies_df: pd.DataFrame,
                         ratings_df: pd.DataFrame) -> pd.DataFrame:
    num_ratings = ratings_df.groupby("movieId")["rating"].count()
    sum_ratings = ratings_df.groupby("movieId")["rating"].sum()
    mean_ratings = ratings_df.groupby("movieId")["rating"].mean()
    global_mean = ratings_df["rating"].mean()

    movies_df["num_ratings"] = movies_df["movieId"].map(num_ratings)
    movies_df["sum_ratings"] = movies_df["movieId"].map(sum_ratings)
    movies_df["mean_ratings"] = movies_df["movieId"].map(mean_ratings)

    damped_mean_ratings = damped_mean(mean_ratings, num_ratings, global_mean, 10)

    movies_df["damped_mean_ratings"] = movies_df["movieId"].map(damped_mean_ratings)

    return movies_df


# a function to convert index to title
def get_title_from_index(movies: pd.DataFrame, index: int):
    return movies[movies.index == index]['title'].values[0]


# a function to convert title to index
def get_index_from_title(movies: pd.DataFrame, title: str):
    return movies[movies.title == title].index.values[0]


# a function to return the most similar title to the words a user type
def find_closest_title(movies: pd.DataFrame, title: str):
    # computes the levenstein scores to each other title, enumerate for matching
    leven_scores = list(enumerate(movies['title'].apply(levenshtein_distance, s2=title)))
    # scores are (idx, score), so sort by score
    sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
    # get the title of the movie with the highest score
    closest_title = get_title_from_index(movies, sorted_leven_scores[0][0])
    # define a variable to store the score of the match ( could probably be removed)
    distance_score = sorted_leven_scores[0][1]

    return closest_title, distance_score
