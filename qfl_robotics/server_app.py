import csv
import os
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

def aggregate_metrics(metrics):
    if not metrics:
        return {}
    
    # Calculate weighted averages based on client dataset sizes
    total_examples = sum([num for num, _ in metrics])
    acc = sum([num * m["accuracy"] for num, m in metrics]) / total_examples
    f1 = sum([num * m["f1_score"] for num, m in metrics]) / total_examples
    
    # Write to CSV
    file_exists = os.path.isfile("benchmark_results.csv")
    with open("benchmark_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Accuracy", "F1_Score"])
        
        model_type = os.environ.get("MODEL_TYPE", "HybridQNN")
        writer.writerow([model_type, acc, f1])
        
    return {"accuracy": acc, "f1_score": f1}

def server_fn(context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )
    config = ServerConfig(num_rounds=8)
    
    # Fixed parameter name: 'config' instead of 'server_config'
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)