
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "electronic-city-traffic-dataset-master"

# Check all traffic matrices
traffic = DATA / "traffic"

for name in [
    "speed_matrix.npy",
    "vehicle_matrix.npy",
    "congestion_matrix.npy",
]:
    path = traffic / name
    print(f"\n--- {name} ---")

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    arr = np.load(path, allow_pickle=False)
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("Min:", float(np.nanmin(arr)))
    print("Max:", float(np.nanmax(arr)))
    print("Non-finite values:", int((~np.isfinite(arr)).sum()))

# Inspect timestamps and graph CSV files
for name in ["timestamps.csv", "nodes_final.csv", "edges_final.csv"]:
    matches = list(DATA.rglob(name))
    print(f"\n--- {name} ---")

    if not matches:
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(matches[0])
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))
