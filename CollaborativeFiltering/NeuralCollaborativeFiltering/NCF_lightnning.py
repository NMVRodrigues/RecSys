import pandas as pd
import numpy as np
import torch
import os
import sys
import random
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.mlense_functions import get_user_split, get_user_item_split


class MovieLensDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
    
    def __getitem__(self, index):
        return {
            'user': self.user_tensor[index],
            'item': self.item_tensor[index],
            'rating': self.target_tensor[index]
        }
    
    def __len__(self):
        return self.user_tensor.size(0)

def prepare_dataset(df, user_encoder=None, item_encoder=None):
    # Dataloader expects IDs to be contiguous integers starting from 0, thus the encoding
    if user_encoder is None:
        user_encoder = LabelEncoder()
        df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
    else:
        # Handle users not seen during training
        df['user_encoded'] = df['user_id'].apply(
            lambda x: user_encoder.transform([x])[0] if x in user_encoder.classes_ else -1
        )
    
    if item_encoder is None:
        item_encoder = LabelEncoder()
        df['item_encoded'] = item_encoder.fit_transform(df['item_id'])
    else:
        # Handle items not seen during training
        df['item_encoded'] = df['item_id'].apply(
            lambda x: item_encoder.transform([x])[0] if x in item_encoder.classes_ else -1
        )
    
    return df, user_encoder, item_encoder

def create_dataloaders(train_data, test_data, batch_size=32):

    train_data, user_encoder, item_encoder = prepare_dataset(train_data)
    
    test_data, _, _ = prepare_dataset(test_data, user_encoder, item_encoder)
    
    # Remove rows with -1 encoding (users/items not seen during training)
    # This only happens when the data-split ensures the same user can't be in both train and test sets
    test_data = test_data[(test_data['user_encoded'] != -1) & (test_data['item_encoded'] != -1)]
    
    train_dataset = MovieLensDataset(
        torch.LongTensor(train_data['user_encoded'].values),
        torch.LongTensor(train_data['item_encoded'].values),
        torch.FloatTensor(train_data['rating'].values)
    )
    
    test_dataset = MovieLensDataset(
        torch.LongTensor(test_data['user_encoded'].values),
        torch.LongTensor(test_data['item_encoded'].values),
        torch.FloatTensor(test_data['rating'].values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, user_encoder, item_encoder

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        x = self.relu(self.fc1(vector))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        prediction = self.output(x)
        return prediction

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    actuals = []
    
    for batch in train_loader:
        user = batch['user'].to(device)
        item = batch['item'].to(device)
        rating = batch['rating'].to(device).float()
        
        # Forward pass
        prediction = model(user, item)
        loss = criterion(prediction.squeeze(), rating)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions.extend(prediction.squeeze().detach().cpu().numpy())
        actuals.extend(rating.detach().cpu().numpy())
    
    return total_loss / len(train_loader), predictions, actuals

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            rating = batch['rating'].to(device).float()
            
            prediction = model(user, item)
            loss = criterion(prediction.squeeze(), rating)
            total_loss += loss.item()
            
            predictions.extend(prediction.squeeze().cpu().numpy())
            actuals.extend(rating.cpu().numpy())
    
    return total_loss / len(test_loader), predictions, actuals

def predict_rating(model, user_id, item_id, device):
    """
    Predict the rating for a given user-item pair
    """
    model.eval()
    with torch.no_grad():
        user = torch.LongTensor([user_id]).to(device)
        item = torch.LongTensor([item_id]).to(device)
        prediction = model(user, item)
        return prediction.item()

def get_top_n_recommendations(model, user_id, n, num_items, device):
    """
    Get top N item recommendations for a user
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        user = torch.LongTensor([user_id] * num_items).to(device)
        items = torch.LongTensor(range(num_items)).to(device)
        predictions = model(user, items).squeeze().cpu().numpy()
    
    # Get indices of top N predictions
    top_n_indices = np.argsort(predictions)[-n:][::-1]
    return list(zip(top_n_indices, predictions[top_n_indices]))


def main():

    parser = argparse.ArgumentParser()

    # data

    parser.add_argument(
        '--type', dest='type', type=str, default='user-item',
        help="If the split should be made on an user or interaction basis")

    parser.add_argument(
        '--seed', dest='seed', type=int, default=42,
        help="Seed for RNG reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data...")
   
    df = pd.read_csv(os.path.join(f'.{os.sep}Datasets', 'ml-100k', 'u.data'), sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    
    if args.type == 'user-item':
        train_ui, test_ui = get_user_item_split(df)
    else:
        train_ui, test_ui = get_user_split(df)
    
    train_loader, test_loader, user_encoder, item_encoder = create_dataloaders(train_ui, test_ui, 128)
    
    print("Preparing the model...")

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    embedding_size = 32
    
    model = NCF(num_users, num_items, embedding_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    num_epochs = 65
    print("Starting training...")
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            train_loss, train_pred, train_actual = train_model(model, train_loader, criterion, optimizer, device)
            test_loss, test_pred, test_actual = evaluate_model(model, test_loader, criterion, device)
        
            train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
            train_mae = mean_absolute_error(train_actual, train_pred)
            test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
            test_mae = mean_absolute_error(test_actual, test_pred)
            
            pbar.set_description(f"Train Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}")
            pbar.update(1)

            # Print metrics for history, TODO: store them on a dict and print the best main metric
            tqdm.write(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {test_loss:.4f}, Val rmse = {test_rmse:.4f}, Val mae = {test_mae:.4f}")


    # Inference examples
    print("\nPerforming Inference:")
    
    # Example 1: Predict rating for a specific user-item pair
    sample_user_id = 42
    sample_item_id = 10
    predicted_rating = predict_rating(model, sample_user_id, sample_item_id, device)
    print(f"Predicted rating for User {sample_user_id} and Item {sample_item_id}: {predicted_rating:.2f}")
    
    # Example 2: Get top 5 recommendations for a user
    sample_user_id = 42
    top_n = 5
    recommendations = get_top_n_recommendations(model, sample_user_id, top_n, num_items, device)
    print(f"\nTop {top_n} recommendations for User {sample_user_id}:")
    for item_id, pred_rating in recommendations:
        print(f"Item {item_id}: Predicted rating {pred_rating:.2f}")

if __name__ == "__main__":
    main()