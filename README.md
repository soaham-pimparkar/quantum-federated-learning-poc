# Quantum Federated Learning (QFL) for Industrial Robotics Predictive Maintenance

A hybrid classical-quantum federated learning framework designed to predict industrial equipment failure (Remaining Useful Life) while maximizing privacy and minimizing network bandwidth overhead on edge nodes.

---

## 🚀 Overview & Problem Statement

In industrial robotics and manufacturing telemetry, predictive maintenance models prevent catastrophic machine failure. However, traditional machine learning approaches face two major bottlenecks:
1. **Data Privacy & Regulation:** Factories are hesitant to centralize raw, high-frequency sensor streams (e.g., vibration, pressure, temperature) to a cloud server due to proprietary operational security risks.
2. **Network Bandwidth Constraints:** Edge devices often operate over constrained cellular or industrial IoT links where transmitting multi-megabyte parameter updates from heavy deep learning models is inefficient.

### How this Proof-of-Concept (POC) Solves It:
* **Federated Learning (Flower Framework):** Raw telemetry never leaves the local edge node. Only lightweight model parameters are aggregated securely via Federated Averaging (`FedAvg`).
* **Quantum Advantage (PennyLane + PyTorch):** By mapping sensor features into a parameterised quantum circuit via **Angle Embedding** and **Entangling Layers**, the model compresses complex multi-variate correlations into a sparse, highly dense state space.
* **Bandwidth Efficiency:** Transmits updates under **$\sim 1.1\text{ KB}$**, offering extreme communication efficiency compared to classical deep learning models.

---

## ⚙️ System Architecture

* **Domain Dataset:** NASA CMAPSS (C-MAPSS) Jet Engine Simulated Data (`train_FD001.txt`), framed as binary classification predicting failure within 30 cycles (`RUL <= 30`).
* **Model Design:** Hybrid Quantum-Classical Neural Network (HQNN).
  * *Classical Input:* PyTorch linear layers mapping 24 sensor features down to 8 qubits.
  * *Quantum Core:* PennyLane 8-qubit circuit utilizing `default.qubit` (optimized for CPU/GPU hybrid handoff).
  * *Output:* Classical sigmoid classification head.
* **Orchestration Framework:** [Flower (`flwr`)](https://flower.ai/) running local simulations via Ray backend.

---

## 🛠️ Installation & Prerequisites

Ensure you have **Python 3.12+** and **uv** installed.

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/quantum-federated-learning.git](https://github.com/YOUR_USERNAME/quantum-federated-learning.git)
   cd quantum-federated-learning