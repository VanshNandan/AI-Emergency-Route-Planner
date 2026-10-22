
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
traffic = root / "data" / "electronic-city-traffic-dataset-master" / "traffic"

for name in ["speed_matrix.npy", "vehicle_matrix.npy", "congestion_matrix.npy"]:
    path = traffic / name
    print(f"\n--- {name} ---")
    print("Exists:", path.exists())

    if not path.exists():
        continue

    with open(path, "rb") as f:
        header = f.read(128)

    print("File size:", path.stat().st_size, "bytes")
    print("Header:", repr(header[:32]))

    try:
        arr = np.load(path, allow_pickle=False)
        print("Shape:", arr.shape)
        print("Dtype:", arr.dtype)
    except Exception as e:
        print("Load error:", type(e).__name__, str(e))
