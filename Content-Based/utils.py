import pandas as pd


def get_ml_ratings_stats(movies_df: pd.DataFrame,
                         ratings_df: pd.DataFrame):
    num_ratings = ratings_df.groupby('movieId')['rating'].count()
    sum_ratings = ratings_df.groupby('movieId')['rating'].sum()
    mean_ratings = ratings_df.groupby('movieId')['rating'].mean()
    global_mean = ratings_df["rating"].mean()

    movies_df['num_ratings'] = movies_df['movieId'].map(num_ratings)
    movies_df['sum_ratings'] = movies_df['movieId'].map(sum_ratings)
    movies_df['mean_ratings'] = movies_df['movieId'].map(mean_ratings)

    damped_mean_ratings = damped_mean(mean_ratings, num_ratings, global_mean, 10)

    movies_df['damped_mean_ratings'] = movies_df['movieId'].map(damped_mean_ratings)

    return movies_df


def damped_mean(
                mean_ratings_name: pd.DataFrame,
                num_ratings_name: pd.DataFrame,
                global_mean_rating: pd.DataFrame,
                damping_factor: int = 5):

    # Compute the damped mean
    damped_mean_ratings = (num_ratings_name * mean_ratings_name + damping_factor * global_mean_rating) \
                          / (num_ratings_name + damping_factor)

    return damped_mean_ratings

