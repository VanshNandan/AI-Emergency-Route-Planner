
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "electronic-city-traffic-dataset-master"

print("DATASET PATH:", DATA)
print("Dataset folder exists:", DATA.exists())

# Inspect CSV files
for filename in ["timestamps.csv", "nodes_final.csv", "edges_final.csv"]:
    matches = list(DATA.rglob(filename))

    if not matches:
        print(f"\nMISSING: {filename}")
        continue

    path = matches[0]
    df = pd.read_csv(path)

    print(f"\n--- {filename} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))

# Inspect NumPy traffic matrices
for filename in [
    "speed_matrix.npy",
    "vehicle_matrix.npy",
    "congestion_matrix.npy",
]:
    matches = list(DATA.rglob(filename))

    if not matches:
        print(f"\nMISSING: {filename}")
        continue

    array = np.load(matches[0], allow_pickle=True)

    print(f"\n--- {filename} ---")
    print("Shape:", array.shape)
    print("Data type:", array.dtype)
    print("Sample values:", array.flat[:5])
    print("Non-finite values:", int((~np.isfinite(array)).sum()))
