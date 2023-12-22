import numpy as np
from scipy.stats import pearsonr
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from copy import deepcopy
import implicit

def weighted_sim_rating(data, sims, target):
    target_ = deepcopy(target)
    for elem in range(len(target_)):
        if target_[elem] == 0:
            rec = np.divide(np.sum(data[:, elem]*sims), np.sum(sims))
            target_[elem] = np.ceil(rec)

    return target_


def memory_cf(data, target, top_sim):

    sim_metric = cosine_similarity
    sim_matrix = sim_metric(data,data)
    target_sim = sim_matrix[target]
    target_sim[target] = 0

    #This approach has a time complexity of O(n + k * log(k)), where n is the number of elements in the list, and k is the number of elements you want to retrieve (in this case, 3).
    similar_entries = [(i,v) for i, v in nlargest(top_sim, enumerate(target_sim), key=lambda x: x[1])]

    ids, similarities = map(list, zip(*similar_entries))

    target_ratings = data[ids]

    recommendations = weighted_sim_rating(target_ratings, similarities, data[target])

    return recommendations

def main():

    data = np.array(
        [
            [2,0,0,4,5,0],
            [5,0,4,0,0,1],
            [0,0,5,0,2,0],
            [0,1,0,5,0,4],
            [0,0,4,0,0,2],
            [4,5,0,1,0,0]
        ]
    )
    '''
    recomendations = []
    for user in range(data.shape[0]):
        recomendations.append(memory_cf(data=data, target=user, top_sim=3))

    print('original')
    print(data, '\n')
    print('after recommendations')
    print(np.array(recomendations))
    '''

    data_ones = np.ones(data.shape[0])
    coo_data = coo_matrix((data_ones, (list(range(data.shape[0])), list(range(data.shape[1])))), shape=(data.shape[0], data.shape[1]))

    model = implicit.als.AlternatingLeastSquares(factors=2,
                                                 iterations=5,
                                                 regularization=0.01,
                                                 random_state=42)
    model.fit(coo_data, show_progress=True)

    csr_data = coo_data.tocsr()
    ids, scores = model.recommend(0, csr_data[0])
    print('ids: ', ids)
    print('scores: ', scores)

if __name__ == '__main__':
    main()
