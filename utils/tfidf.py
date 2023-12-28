import itertools
from typing import List

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ._stop_words import ENGLISH_STOP_WORDS


def tfidf(documents: List[str],
          document_words: List[List[str]],
          norm: bool = False) -> NDArray[[float]]:

    """
    Simple naive implementation of tfidf. Output differ from scikit-learn, probably due ot the smoothing
    :param documents: List of document names or Uids
    :param document_words: List of lists of words. Each list of words corresponds to the contents of a doc
    :param norm: Weather to apply L2 (euclidean) norm or not
    :return:
    """

    # get number of unique words for indexing and building placeholder matrix
    unique_words = list(
        np.unique(np.array(list(itertools.chain.from_iterable(document_words))))
    )
    # build placeholder matrix
    tfidf_matrix = np.zeros((len(documents), len(unique_words)))

    # make the dictionary
    document_dict = {}
    # basically a df lookup table
    word_dict = {}
    for doc, wl in zip(documents, document_words):
        # remove stop words
        filtered_wl = list(set(wl).difference(ENGLISH_STOP_WORDS))
        document_dict[doc] = filtered_wl

    for word in unique_words:
        word_dict[word] = sum(word in document for document in document_dict.values())

    N = len(list(document_dict.keys()))

    for doc_idx, doc in enumerate(document_dict.keys()):
        for word in document_dict[doc]:
            df = word_dict[word]
            f = document_dict[doc].count(word)

            idf = np.log((1 + N) / (1 + df)) + 1
            tf = f / len(document_dict[doc])

            word_idx = unique_words.index(word)

            tfidf_matrix[doc_idx][word_idx] = tf * idf

    if norm is True:
        tfidf_matrix = tfidf_matrix / np.linalg.norm(
            tfidf_matrix, axis=1, keepdims=True
        )

    return tfidf_matrix