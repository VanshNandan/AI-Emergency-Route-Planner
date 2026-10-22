
from pathlib import Path
import heapq

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "electronic-city-traffic-dataset-master"
TRAFFIC = DATA / "traffic"
GRAPH = DATA / "graph"
MODELS = ROOT / "models"

speed = np.load(TRAFFIC / "speed_matrix.npy", allow_pickle=False)
vehicles = np.load(TRAFFIC / "vehicle_matrix.npy", allow_pickle=False)
congestion = np.load(TRAFFIC / "congestion_matrix.npy", allow_pickle=False)
timestamps = pd.to_datetime(
    pd.read_csv(TRAFFIC / "timestamps.csv")["timestamp"]
)

nodes = pd.read_csv(GRAPH / "nodes_final.csv")
edges = pd.read_csv(GRAPH / "edges_final.csv")
model = joblib.load(MODELS / "congestion_model.joblib")

# Verify that edge IDs match traffic-matrix column indices.
edges["edge_id"] = pd.to_numeric(edges["edge_id"], errors="raise")
edge_ids = sorted(edges["edge_id"].astype(int).tolist())

if edge_ids != list(range(congestion.shape[1])):
    raise ValueError(
        "Edge IDs do not match traffic matrix columns. "
        "Do not run routing until the dataset's edge ordering is verified."
    )

edges = edges.sort_values("edge_id").reset_index(drop=True)
node_ids = set(nodes["node_id"].astype(int))

if not set(edges["u"].astype(int)).issubset(node_ids):
    raise ValueError("Some edge start nodes are missing from nodes_final.csv.")
if not set(edges["v"].astype(int)).issubset(node_ids):
    raise ValueError("Some edge end nodes are missing from nodes_final.csv.")

# Choose the latest available traffic snapshot.
T = len(timestamps) - 1
previous_t = max(T - 1, 0)
current_time = timestamps.iloc[T]

# Predict congestion 5 minutes after the final observed snapshot.
features = np.column_stack([
    speed[T],
    vehicles[T],
    congestion[T],
    congestion[previous_t],
    np.full(congestion.shape[1], current_time.hour),
    np.full(congestion.shape[1], current_time.dayofweek),
    np.arange(congestion.shape[1]),
]).astype(np.float32)

predicted_congestion = np.clip(model.predict(features), 0, 1)

def parse_speed(value):
    """Parse speed limits such as '40', '50 km/h', or 'signals'."""
    if pd.isna(value):
        return 40.0
    import re
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else 40.0

def edge_travel_time(edge):
    """Estimated edge time in minutes, adjusted for predicted congestion."""
    edge_id = int(edge.edge_id)
    length_km = max(float(edge.length_m), 1.0) / 1000.0
    limit = parse_speed(edge.maxspeed)

    observed_speed = float(speed[T, edge_id])
    # Do not exceed the road limit when one is available.
    base_speed = max(5.0, min(observed_speed, limit))
    congestion_factor = 1.0 + 2.0 * float(predicted_congestion[edge_id])
    effective_speed = max(3.0, base_speed / congestion_factor)

    return length_km / effective_speed * 60.0

# Build directed adjacency list.
adjacency = {int(node): [] for node in nodes["node_id"]}
for edge in edges.itertuples(index=False):
    adjacency[int(edge.u)].append(
        (int(edge.v), edge_travel_time(edge), int(edge.edge_id))
    )

# A* heuristic: straight-line distance divided by the maximum dataset speed.
node_lookup = nodes.set_index("node_id")
MAX_SPEED_KMH = 80.0

def heuristic(a, b):
    ra = node_lookup.loc[a]
    rb = node_lookup.loc[b]
    # x and y are longitude and latitude, so approximate distance in km.
    mean_lat = np.radians((float(ra.y) + float(rb.y)) / 2)
    dx = (float(rb.x) - float(ra.x)) * 111.32 * np.cos(mean_lat)
    dy = (float(rb.y) - float(ra.y)) * 111.32
    distance_km = np.hypot(dx, dy)
    return distance_km / MAX_SPEED_KMH * 60.0

def astar(start, goal):
    start, goal = int(start), int(goal)

    if start not in adjacency or goal not in adjacency:
        raise ValueError("Start or destination node does not exist.")

    queue = [(heuristic(start, goal), 0.0, start)]
    best_cost = {start: 0.0}
    parent = {}

    while queue:
        _, cost, current = heapq.heappop(queue)

        if cost > best_cost.get(current, float("inf")):
            continue

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]][0])
            path.reverse()

            path_edges = [
                parent[path[i]][1] for i in range(1, len(path))
            ]
            return path, path_edges, cost

        for neighbor, travel_time, edge_id in adjacency[current]:
            new_cost = cost + travel_time
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = (current, edge_id)
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, new_cost, neighbor))

    return None, [], float("inf")

def find_route(start, goal):
    path, path_edges, minutes = astar(start, goal)
    if path is None:
        return {
            "start": int(start),
            "destination": int(goal),
            "reachable": False,
            "nodes": [],
            "edges": [],
            "estimated_minutes": None,
        }

    return {
        "start": int(start),
        "destination": int(goal),
        "reachable": True,
        "nodes": path,
        "edges": path_edges,
        "estimated_minutes": round(minutes, 2),
    }

if __name__ == "__main__":
    print("Available nodes:", len(node_ids))
    print("Available directed edges:", len(edges))
    print("Traffic snapshot:", current_time)
    print("Predicted congestion range:",
          round(float(predicted_congestion.min()), 3),
          "to",
          round(float(predicted_congestion.max()), 3))

    # Try a route between two valid nodes.
    result = find_route(0, 100)
    print("\nRoute result:")
    print(result)
