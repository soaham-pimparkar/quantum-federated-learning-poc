import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

def load_cmapss_data(client_id, num_clients=5, file_path="datasets/CMAPSSData/train_FD001.txt"):
    """
    Loads CMAPSS FD001, scales features, frames as binary classification (RUL <= 30),
    and distributes engines across federated clients.
    """
    # 1. Define CMAPSS columns (26 columns total)
    columns = ['unit', 'time_cycles']
    columns += [f'op_setting_{i}' for i in range(1, 4)]
    columns += [f'sensor_{i}' for i in range(1, 22)]
    
    # 2. Load the data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)
    df = pd.read_csv(full_path, sep=r'\s+', header=None, names=columns)
    
    # 3. Calculate Remaining Useful Life (RUL) per unit
    rul = pd.DataFrame(df.groupby('unit')['time_cycles'].max()).reset_index()
    rul.columns = ['unit', 'max_cycles']
    df = df.merge(rul, on=['unit'], how='left')
    df['RUL'] = df['max_cycles'] - df['time_cycles']
    df.drop('max_cycles', axis=1, inplace=True)
    
    # 4. Binary Classification Threshold (Failure within 30 cycles)
    df['label'] = np.where(df['RUL'] <= 30, 1, 0)
    
    # 5. Split Engines (Units) among clients
    unique_units = df['unit'].unique()
    units_per_client = len(unique_units) // num_clients
    start_idx = client_id * units_per_client
    end_idx = start_idx + units_per_client if client_id != num_clients - 1 else len(unique_units)
    
    client_units = unique_units[start_idx:end_idx]
    client_df = df[df['unit'].isin(client_units)].copy()
    
    # 6. Feature Selection & Scaling (Crucial for Quantum Angle Embedding)
    features = [c for c in columns if c not in ['unit', 'time_cycles']]
    
    scaler = MinMaxScaler(feature_range=(0, np.pi)) # Scale to [0, Pi] for quantum rotations
    X_scaled = scaler.fit_transform(client_df[features])
    y = client_df['label'].values
    
    # 7. Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 8. Train/Test Split (80/20 for local evaluation)
    split_idx = int(len(X_tensor) * 0.8)
    train_ds = TensorDataset(X_tensor[:split_idx], y_tensor[:split_idx])
    test_ds = TensorDataset(X_tensor[split_idx:], y_tensor[split_idx:])
    
    # Create DataLoaders (Batch size 64 for faster processing of thousands of rows)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    return train_loader, test_loader