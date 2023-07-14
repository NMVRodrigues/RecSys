import numpy as np
from scipy.stats import pearsonr

def user_cf(data, target):
    users = {idx:items for idx,items in enumerate(data)}
    target_user = users[target]
    sim_metric = np.corrcoef


    print(users[2])
    print(users[4])
    print(sim_metric(users[4], users[2]))

def main():

    # Data example taken from https://www.youtube.com/watch?v=bLhq63ygoU8&ab_channel=AlexSmola
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

    user_cf(data, 4)

if __name__ == '__main__':
    main()
