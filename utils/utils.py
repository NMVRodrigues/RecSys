import itertools

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


def damped_mean(
    mean_ratings_name: pd.DataFrame,
    num_ratings_name: pd.DataFrame,
    global_mean_rating: pd.DataFrame,
    damping_factor: int = 5,
) -> pd.DataFrame:
    # Compute the damped mean
    damped_mean_ratings = (
        num_ratings_name * mean_ratings_name + damping_factor * global_mean_rating
    ) / (num_ratings_name + damping_factor)

    return damped_mean_ratings


def levenshtein_distance_manual(s1: str, s2: str):
    """
    Calculates the Levenshtein distance between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """

    m = len(s1)
    n = len(s2)

    # Initialize the matrix, +1 fpr empty strings
    dp = np.zeros((m + 1, n + 1),dtype=int)

    # Fill the matrix in a bottom-up fashion
    # compare each char of string M to each char of string N
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i, j] = j
            elif j == 0:
                dp[i, j] = i
            elif s1[i - 1] != s2[j - 1]:
                dp[i, j] = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + 1)
            else:
                dp[i, j] = dp[i - 1, j - 1]

    return dp[m, n]


def levenshtein_distance(s1: str, s2: str):
    # ratio [0-100], 100 perfect match
    return fuzz.ratio(s1, s2)

