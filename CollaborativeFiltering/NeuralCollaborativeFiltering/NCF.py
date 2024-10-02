import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv(os.path.join(f'.{os.sep}Datasets', 'ml-100k', 'u.data'), sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
print(df.head())

'''
# Custom dataset class
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

# NCF Model (same as before)
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

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
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
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            rating = batch['rating'].to(device).float()
            
            prediction = model(user, item)
            loss = criterion(prediction.squeeze(), rating)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    print("Loading data...")
    df = get_movielens_data()
    
    # Convert ratings to binary (1 if rating >= 4, 0 otherwise)
    df['rating'] = (df['rating'] >= 4).astype(int)
    
    # Encode user and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['item_encoded'] = item_encoder.fit_transform(df['item_id'])
    
    # Split data
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create tensors
    train_user_tensor = torch.LongTensor(train_data['user_encoded'].values)
    train_item_tensor = torch.LongTensor(train_data['item_encoded'].values)
    train_rating_tensor = torch.FloatTensor(train_data['rating'].values)
    
    test_user_tensor = torch.LongTensor(test_data['user_encoded'].values)
    test_item_tensor = torch.LongTensor(test_data['item_encoded'].values)
    test_rating_tensor = torch.FloatTensor(test_data['rating'].values)
    
    # Create datasets and dataloaders
    train_dataset = MovieLensDataset(train_user_tensor, train_item_tensor, train_rating_tensor)
    test_dataset = MovieLensDataset(test_user_tensor, test_item_tensor, test_rating_tensor)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    embedding_size = 32
    
    model = NCF(num_users, num_items, embedding_size).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 5
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    main()

'''