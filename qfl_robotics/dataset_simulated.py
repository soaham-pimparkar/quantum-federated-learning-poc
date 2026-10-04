import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_simulated_robot_data(client_id: int, num_samples: int = 200, batch_size: int = 16):
    """Generates synthetic kinematic telemetry for multi-vendor robot arms.
    Each client_id gets slightly different data distributions (Non-IID).
    """
    np.random.seed(client_id + 42)
    torch.manual_seed(client_id + 42)
    
    # 8 kinematic features: [torque_j1, torque_j2, speed_j1, speed_j2, vib_x, vib_y, temp_m1, temp_m2]
    shift = client_id * 0.5  # Simulate multi-vendor calibration variance
    X = np.random.randn(num_samples, 8) + shift
    
    # Target: 0 = Normal operation, 1 = Kinematic degradation warning
    # Synthetic rule based on non-linear combination of torque and vibration
    y = ((X[:, 0] * X[:, 4] + X[:, 1] * X[:, 5]) > 0.5).astype(np.int64)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 80/20 train/test split
    split = int(0.8 * num_samples)
    train_ds = TensorDataset(X_tensor[:split], y_tensor[:split])
    test_ds = TensorDataset(X_tensor[split:], y_tensor[split:])
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)