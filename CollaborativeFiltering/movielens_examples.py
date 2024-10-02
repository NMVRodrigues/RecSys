from fastai.collab import *
from fastai.tabular.all import *
from scipy.sparse import coo_matrix

set_seed(42)

# Specify the path to the locally downloaded dataset file
file_path = 'C:\\Users\\Utilizador\\Desktop\\RecSys\\Datasets\ml-100k\\u.data'

data = pd.read_csv(file_path, delimiter='\t', header=None, names=['user','movie','rating','timestamp'])


# get user and item ids into lists
ALL_USERS = data['user'].unique().tolist()
ALL_ITEMS = data['movie'].unique().tolist()

# make a mapping between id - int [0-all]
user_ids = dict(list(enumerate(ALL_USERS)))
item_ids = dict(list(enumerate(ALL_ITEMS)))

# iverts the mapping to be int [0-all] - id
user_map = {u: uidx for uidx, u in user_ids.items()}
item_map = {i: iidx for iidx, i in item_ids.items()}

print(user_map)

user_map = {i: ALL_USERS[i] for i in range(len(ALL_USERS))}
item_map = {i: ALL_ITEMS[i] for i in range(len(ALL_ITEMS))}

print(user_map)

# creates two new columns, mapping each customer and article id into their int [0-all]
data['user_id'] = data['user'].map(user_map)
data['item_id'] = data['movie'].map(item_map)

row = data['user_id'].values
col = data['item_id'].values

#  shape is number of rows, so number of interactions user x item
ones = np.ones(data.shape[0])

print(f'row shape {row.shape}')
print(f'col shape {col.shape}')
print(f'data shape {ones.shape}')
print(f'USERS shape {len(ALL_ITEMS)}')
print(f'ITEMS shape {len(ALL_USERS)}')

# shape needs to take into account unique users and items, not duplicates
coo_train = coo_matrix((ones, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))

print(coo_matrix)