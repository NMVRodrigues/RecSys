import pandas
import pandas as pd
import os

from .utils import damped_mean, levenshtein_distance


def prepare_ml(folder: str) -> (pd.DataFrame, pd.DataFrame):
    movies = pd.read_csv(os.path.join(folder, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(folder, 'ratings.csv'))

    return movies, ratings


def fix_ml(movies:pd.DataFrame) -> pd.DataFrame:

    # create new column for the years
    movies['year'] = [0]*movies.index

    for ind in movies.index:
        #movies['year'][ind] = movies['title'][ind].split(' (')[-1][:-1]
        #movies['title'][ind] = movies['title'][ind].split(' (')[0]
        movies.loc[ind, 'year'] = movies['title'][ind].split(' (')[-1][:-1]
        movies.loc[ind, 'title'] = movies['title'][ind].split(' (')[0]

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
    leven_scores = list(enumerate(movies['title'].apply(levenshtein_distance, s2=title)))
    sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
    closest_title = get_title_from_index(movies, sorted_leven_scores[0][0])
    distance_score = sorted_leven_scores[0][1]

    return closest_title, distance_score
