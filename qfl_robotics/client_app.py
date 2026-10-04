import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from sklearn.metrics import f1_score

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
# from qfl_robotics.dataset_simulated import load_simulated_robot_data
from qfl_robotics.dataset import load_cmapss_data

# 1. Hybrid Hardware Allocation
# PyTorch uses the RTX 3060, PennyLane uses the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

# 2. Quantum Circuit & Hybrid Model
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(0))]

class ClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the 24 inputs and the hidden dimensions of the QNN
        self.layers = nn.Sequential(
            nn.Linear(24, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class HybridQNN(nn.Module):
    def __init__(self, n_layers=3): # Increased entanglement depth
        super().__init__()
        self.clayer_in = nn.Linear(24, n_qubits) # Compress 24 sensors to 8 qubits
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.clayer_out = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.tanh(self.clayer_in(x))
        x = self.qlayer(x)
        x = self.clayer_out(x)
        return self.sigmoid(x)

# 3. Payload Utility Function
def print_payload_size(parameters, client_id):
    """Calculates the true byte size of the numpy arrays being sent over the network."""
    total_bytes = sum(arr.nbytes for arr in parameters)
    kb_size = total_bytes / 1024.0
    print(f"\n[Client {client_id}] Transmitting {len(parameters)} parameter tensors.")
    print(f"[Client {client_id}] Total Network Payload Size: {kb_size:.4f} KB\n")

# 4. Flower Client
class QFLClient(NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id):
        self.model = model.to(device)
        self.device = device
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.criterion = nn.BCELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        # Increased to 5 local epochs
        for epoch in range(5): 
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
        updated_params = self.get_parameters(config={})
        # Calculate and print the payload size before sending to the server
        print_payload_size(updated_params, self.client_id)
        
        return updated_params, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss += self.criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                
                # Store for F1 calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate F1 Score
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        
        return float(loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": float(accuracy), "f1_score": float(f1)}


def client_fn(context: Context):
    client_id = context.node_config.get("partition-id", 0)
    train_loader, test_loader = load_cmapss_data(client_id=client_id, num_clients=5)
    
    # Dynamically select the model based on environment variable
    model_type = os.environ.get("MODEL_TYPE", "HybridQNN")
    if model_type == "MLP":
        model = ClassicalMLP()
    else:
        model = HybridQNN(n_layers=3) # Using the 8-qubit, 3-layer version
        
    return QFLClient(model, train_loader, test_loader, device, client_id).to_client()

app = ClientApp(client_fn=client_fn)