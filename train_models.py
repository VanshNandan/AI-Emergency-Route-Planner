
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "electronic-city-traffic-dataset-master"
TRAFFIC = DATA / "traffic"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading traffic data...")

speed = np.load(TRAFFIC / "speed_matrix.npy", allow_pickle=False)
vehicles = np.load(TRAFFIC / "vehicle_matrix.npy", allow_pickle=False)
congestion = np.load(
    TRAFFIC / "congestion_matrix.npy", allow_pickle=False
)
timestamps = pd.to_datetime(
    pd.read_csv(TRAFFIC / "timestamps.csv")["timestamp"]
)

n_times, n_edges = congestion.shape
assert speed.shape == vehicles.shape == congestion.shape
assert len(timestamps) == n_times

# Predict congestion at the NEXT 5-minute timestep.
split = int(n_times * 0.8)
rng = np.random.default_rng(42)

def make_samples(time_indices, count):
    t = rng.choice(time_indices, size=count, replace=True)
    e = rng.integers(0, n_edges, size=count)

    previous_t = np.maximum(t - 1, 0)
    ts = timestamps.iloc[t]

    X = np.column_stack([
        speed[t, e],
        vehicles[t, e],
        congestion[t, e],
        congestion[previous_t, e],
        ts.dt.hour.to_numpy(),
        ts.dt.dayofweek.to_numpy(),
        e,
    ])

    y = congestion[t + 1, e]
    return X.astype(np.float32), y.astype(np.float32)

# Leave the final 20% of the timeline for testing.
X_train, y_train = make_samples(
    np.arange(0, split - 1), 60000
)
X_test, y_test = make_samples(
    np.arange(split, n_times - 1), 15000
)

models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=12, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    ),
}

results = {}
trained = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results[name] = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
        "training_seconds": round(time.time() - start, 2),
    }
    trained[name] = model

    print(results[name])

best_name = min(results, key=lambda name: results[name]["MAE"])
best_model = trained[best_name]

joblib.dump(best_model, MODEL_DIR / "congestion_model.joblib")

with open(MODEL_DIR / "model_metrics.json", "w") as f:
    json.dump(
        {
            "target": "Congestion index 5 minutes ahead",
            "features": [
                "current_speed_kmh",
                "current_vehicle_count",
                "current_congestion",
                "previous_congestion",
                "hour",
                "day_of_week",
                "edge_id",
            ],
            "chronological_split": "First 80% train; final 20% test",
            "sample_sizes": {
                "train": len(y_train),
                "test": len(y_test),
            },
            "results": results,
            "best_model": best_name,
        },
        f,
        indent=2,
    )

print("\n--- MODEL COMPARISON ---")
print(pd.DataFrame(results).T)
print(f"\nBest model by test MAE: {best_name}")
print("Saved model to:", MODEL_DIR / "congestion_model.joblib")
print("Saved metrics to:", MODEL_DIR / "model_metrics.json")
