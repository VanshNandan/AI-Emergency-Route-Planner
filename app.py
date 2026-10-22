
import streamlit as st
import pandas as pd
import numpy as np
import json

from route_planner import (
    find_route,
    nodes,
    edges,
    speed,
    vehicles,
    congestion,
    predicted_congestion,
    timestamps,
)

st.set_page_config(
    page_title="AI Emergency Route Planner",
    page_icon="🚑",
    layout="wide",
)

st.title("🚑 AI-Powered Emergency Route Planner")
st.caption(
    "Machine-learning congestion prediction + A* route optimization"
)

st.info(
    "Academic prototype using synthetically generated Electronic City "
    "traffic data. Results are simulated estimates, not live ambulance ETAs."
)

# Sidebar controls
st.sidebar.header("Route Configuration")

node_ids = sorted(nodes["node_id"].astype(int).tolist())

start = st.sidebar.selectbox(
    "Ambulance starting node",
    node_ids,
    index=node_ids.index(0),
)

destination = st.sidebar.selectbox(
    "Emergency destination node",
    node_ids,
    index=node_ids.index(100),
)

if start == destination:
    st.warning("Please select different start and destination nodes.")
    st.stop()

if st.sidebar.button("Find Emergency Route", type="primary"):
    st.session_state["route_result"] = find_route(start, destination)

result = st.session_state.get("route_result")

# Dataset overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Road Network Nodes", len(nodes))
c2.metric("Directed Road Segments", len(edges))
c3.metric("Traffic Observations", f"{len(timestamps):,}")
c4.metric("Prediction Interval", "5 minutes")

st.divider()

if result is None:
    result = find_route(start, destination)

if result["reachable"]:
    route_nodes = result["nodes"]
    route_edges = result["edges"]

    st.subheader("Recommended Emergency Route")

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Travel Time", f'{result["estimated_minutes"]:.2f} min')
    m2.metric("Nodes in Route", len(route_nodes))
    m3.metric("Road Segments", len(route_edges))

    st.write("**Route sequence**")
    st.code(" → ".join(map(str, route_nodes)))

    # Show details for each selected road segment
    edge_rows = edges.set_index("edge_id").loc[route_edges].reset_index()
    edge_rows["Predicted Congestion"] = [
        float(predicted_congestion[int(e)]) for e in route_edges
    ]
    edge_rows["Current Speed (km/h)"] = [
        float(speed[-1, int(e)]) for e in route_edges
    ]
    edge_rows["Current Vehicle Count"] = [
        float(vehicles[-1, int(e)]) for e in route_edges
    ]

    display_cols = [
        "edge_id", "u", "v", "length_m", "highway",
        "Predicted Congestion", "Current Speed (km/h)",
        "Current Vehicle Count",
    ]
    st.subheader("Road-by-Road Analysis")
    st.dataframe(
        edge_rows[display_cols].round(3),
        use_container_width=True,
        hide_index=True,
    )

    # Geographic route visualization
    node_lookup = nodes.set_index("node_id")
    route_points = node_lookup.loc[route_nodes]

    st.subheader("Route Map")
    st.map(
        pd.DataFrame({
            "latitude": route_points["y"].to_numpy(),
            "longitude": route_points["x"].to_numpy(),
        }),
        latitude="latitude",
        longitude="longitude",
        size=30,
    )

    st.caption(
        "The map plots route nodes geographically. It connects points visually "
        "and does not represent turn-by-turn road geometry."
    )

    # Model metrics saved during training
    metrics_path = "models/model_metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        st.subheader("Machine Learning Evaluation")
        st.write(f'Best model: **{metrics["best_model"]}**')

        metric_df = pd.DataFrame(metrics["results"]).T
        st.dataframe(metric_df.round(4), use_container_width=True)

    except (FileNotFoundError, KeyError, ValueError):
        st.warning("Model evaluation metrics could not be loaded.")

    csv = edge_rows[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Route Details (CSV)",
        data=csv,
        file_name="emergency_route_details.csv",
        mime="text/csv",
    )

else:
    st.error(
        f"No directed route exists from node {start} to node {destination}. "
        "Try another destination."
    )

st.divider()
st.caption(
    "Research prototype | Random Forest congestion prediction | "
    "A* graph search | Python + Streamlit"
)
