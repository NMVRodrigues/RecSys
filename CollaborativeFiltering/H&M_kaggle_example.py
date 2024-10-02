import os
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix
from implicit.evaluation import mean_average_precision_at_k

pd.set_option('display.max_columns', None)

# paths to datasets
dset_path = '../Datasets/H-and-M'
csv_train = os.path.join(dset_path,'transactions_train.csv')
csv_sub = os.path.join(dset_path,'sample_submission.csv')
csv_users = os.path.join(dset_path,'customers.csv')
csv_items = os.path.join(dset_path,'articles.csv')

# read all datasets
df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])
df_sub = pd.read_csv(csv_sub)
dfu = pd.read_csv(csv_users)
dfi = pd.read_csv(csv_items, dtype={'article_id': str})

print(df.head())

# get user and item ids into lists
ALL_USERS = dfu['customer_id'].unique().tolist()
ALL_ITEMS = dfi['article_id'].unique().tolist()

print(ALL_USERS[:10])

# make a mapping between id - int [0-all]
user_ids = dict(list(enumerate(ALL_USERS)))
item_ids = dict(list(enumerate(ALL_ITEMS)))

print(list(user_ids.items())[:10])
# iverts the mapping to be int [0-all] - id
user_map = {u: uidx for uidx, u in user_ids.items()}
item_map = {i: iidx for iidx, i in item_ids.items()}

print(list(user_map.items())[:10])
# creates two new columns, mapping each customer and article id into their int [0-all]
df['user_id'] = df['customer_id'].map(user_map)
df['item_id'] = df['article_id'].map(item_map)

print(df.head())
del dfu, dfi


row = df['user_id'].values
col = df['item_id'].values

#  shape is number of rows, so number of interactions user x item
data = np.ones(df.shape[0])

print(f'row shape {row.shape}')
print(f'col shape {col.shape}')
print(f'data shape {data.shape}')
print(f'USERS shape {len(ALL_ITEMS)}')
print(f'ITEMS shape {len(ALL_USERS)}')

# shape needs to take into account unique users and items, not duplicates
coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))



def to_user_item_coo(df):
    """ Turn a dataframe with transactions into a COO sparse items x users matrix"""
    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    return coo


def split_data(df, validation_days=7):
    """ Split a pandas dataframe into training and validation data, using <<validation_days>>
    """
    validation_cut = df['t_dat'].max() - pd.Timedelta(validation_days)

    df_train = df[df['t_dat'] < validation_cut]
    df_val = df[df['t_dat'] >= validation_cut]
    return df_train, df_val


def get_val_matrices(df, validation_days=7):
    """ Split into training and validation and create various matrices

        Returns a dictionary with the following keys:
            coo_train: training data in COO sparse format and as (users x items)
            csr_train: training data in CSR sparse format and as (users x items)
            csr_val:  validation data in CSR sparse format and as (users x items)

    """
    df_train, df_val = split_data(df, validation_days=validation_days)
    coo_train = to_user_item_coo(df_train)
    coo_val = to_user_item_coo(df_val)

    csr_train = coo_train.tocsr()
    csr_val = coo_val.tocsr()

    return {'coo_train': coo_train,
            'csr_train': csr_train,
            'csr_val': csr_val
            }


def validate(matrices, factors=200, iterations=20, regularization=0.01, show_progress=True):
    """ Train an ALS model with <<factors>> (embeddings dimension)
    for <<iterations>> over matrices and validate with MAP@12
    """
    coo_train, csr_train, csr_val = matrices['coo_train'], matrices['csr_train'], matrices['csr_val']

    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 iterations=iterations,
                                                 regularization=regularization,
                                                 random_state=42)
    model.fit(coo_train, show_progress=show_progress)

    # The MAPK by implicit doesn't allow to calculate allowing repeated items, which is the case.
    # TODO: change MAP@12 to a library that allows repeated items in prediction
    map12 = mean_average_precision_at_k(model, csr_train, csr_val, K=12, show_progress=show_progress, num_threads=4)
    print(
        f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@12: {map12:6.5f}")
    return map12


matrices = get_val_matrices(df)

best_map12 = 0
for factors in [500, 1000, 2000]:
    for iterations in [3, 12, 14, 15, 20]:
        for regularization in [0.01, 0.001, 0.05, 0.024]:
            map12 = validate(matrices, factors, iterations, regularization, show_progress=False)
            if map12 > best_map12:
                best_map12 = map12
                best_params = {'factors': factors, 'iterations': iterations, 'regularization': regularization}
                print(f"Best MAP@12 found. Updating: {best_params}")

del matrices

coo_train = to_user_item_coo(df)
csr_train = coo_train.tocsr()

def train(coo_train, factors=200, iterations=15, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 iterations=iterations,
                                                 regularization=regularization,
                                                 random_state=42)
    model.fit(coo_train, show_progress=show_progress)
    return model

model = train(coo_train, **best_params)


def submit(model, csr_train, submission_name="submissions.csv"):
    preds = []
    batch_size = 2000
    to_generate = np.arange(len(ALL_USERS))
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx: startidx + batch_size]
        ids, scores = model.recommend(batch, csr_train[batch], N=12, filter_already_liked_items=False)
        for i, userid in enumerate(batch):
            customer_id = user_ids[userid]
            user_items = ids[i]
            article_ids = [item_ids[item_id] for item_id in user_items]
            preds.append((customer_id, ' '.join(article_ids)))

    df_preds = pd.DataFrame(preds, columns=['customer_id', 'prediction'])
    df_preds.to_csv(submission_name, index=False)

    print(df_preds.shape)

    return df_preds

df_preds = submit(model, csr_train)