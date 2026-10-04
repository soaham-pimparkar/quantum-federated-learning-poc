import subprocess
import os

print("========================================")
print("  STARTING CLASSICAL MLP BENCHMARK")
print("========================================")
env = os.environ.copy()
env["MODEL_TYPE"] = "MLP"
subprocess.run(["flwr", "run", ".", "--stream"], env=env)

print("\n========================================")
print("  STARTING HYBRID QNN BENCHMARK")
print("========================================")
env["MODEL_TYPE"] = "HybridQNN"
subprocess.run(["flwr", "run", ".", "--stream"], env=env)

print("\n[SUCCESS] Benchmarks complete! Open 'benchmark_results.csv' to see the data.")