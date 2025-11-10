import json
import matplotlib.pyplot as plt
from collections import defaultdict

with open("experiments/all_missing_modality.json", "r") as f:
    data = json.load(f)

def compute_avg_performance(stage_data):
    combinations = stage_data["all_combinations"]
    results = []
    for combo, metrics in combinations.items():
        n_sensors = len(combo.split("+"))
        results.append({
            "n_sensors": n_sensors,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"]
        })
    acc_by_n = defaultdict(list)
    f1_by_n = defaultdict(list)
    for r in results:
        acc_by_n[r["n_sensors"]].append(r["accuracy"])
        f1_by_n[r["n_sensors"]].append(r["f1_macro"])
    avg_n = sorted(acc_by_n.keys())
    avg_acc = [sum(acc_by_n[n]) / len(acc_by_n[n]) for n in avg_n]
    avg_f1 = [sum(f1_by_n[n]) / len(f1_by_n[n]) for n in avg_n]
    return avg_n, avg_acc, avg_f1

stages = ["early", "hybrid", "late"]
colors = ["r", "g", "b"]
stage_results = {}

for stage in stages:
    if stage in data:
        avg_n, avg_acc, avg_f1 = compute_avg_performance(data[stage])
        stage_results[stage] = (avg_n, avg_acc, avg_f1)

plt.figure(figsize=(8, 5))
for stage, color in zip(stage_results.keys(), colors):
    avg_n, avg_acc, avg_f1 = stage_results[stage]
    print(avg_acc[3] - avg_acc[0])
    plt.plot(avg_n, avg_acc, marker="o", linestyle="-", color=color, label=f"{stage.capitalize()} Accuracy")
    plt.plot(avg_n, avg_f1, marker="s", linestyle="--", color=color, label=f"{stage.capitalize()} F1 Macro")

plt.xlabel("Number of Available Sensors")
plt.ylabel("Performance")
plt.title("Degradation Curve: Performance vs Available Sensors")
plt.xticks(sorted(stage_results[list(stage_results.keys())[0]][0]))
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("degradation_curve.png", dpi=300)
