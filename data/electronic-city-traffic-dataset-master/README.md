# Electronic City Traffic Dataset

This repository contains the traffic time-series and road-network data used in the study:

**A Hybrid Deep Learning Framework for Predictive Traffic Congestion Management and Adaptive Route Optimization**

## Dataset Overview

The dataset represents a simulated traffic environment for Electronic City, Bengaluru, India.

The traffic time series was synthetically generated to reproduce realistic temporal and spatial traffic patterns. It is **not derived from real-world traffic sensors**.

The dataset contains:
- 60 days of traffic time-series data
- 5-minute temporal resolution
- 17,280 timestamps
- 602 directed road segments
- Vehicle speed
- Vehicle count
- Congestion index

The road network contains:
- 262 nodes
- 602 directed edges

## Directory Structure

```text
electronic-city-traffic-dataset/
├── traffic/
│   ├── speed_matrix.npy
│   ├── vehicle_matrix.npy
│   ├── congestion_matrix.npy
│   └── timestamps.csv
├── graph/
│   ├── nodes_final.csv
│   ├── edges_final.csv
│   └── electronic_city_final.graphml
└── README.md
```

## Traffic Data

The three NumPy matrices contain traffic values for the 602 directed road segments across the 17,280 timestamps.

### speed_matrix.npy
Vehicle speed for each road segment and timestamp, in km/h.

### vehicle_matrix.npy
Simulated vehicle-count values for each road segment and timestamp.

### congestion_matrix.npy
Congestion-index values for each road segment and timestamp.

### timestamps.csv
Timestamp corresponding to each traffic observation. The temporal resolution is 5 minutes.

## Road Network

The graph files describe the Electronic City road network used in the experiments.

### nodes_final.csv
Contains the network nodes and their geographic coordinates.

### edges_final.csv
Contains the directed road segments, including edge identifiers, connectivity, road length, highway type, and speed-limit information where available.

### electronic_city_final.graphml
NetworkX/OSMnx-compatible representation of the road network.

## Data Generation

The traffic time series was generated synthetically using temporal and spatial traffic patterns, including daily variations, weekend effects, rainfall events, incidents, road-specific factors, and spatial congestion effects.

Because the traffic data are simulated, the dataset should not be interpreted as a record of actual traffic conditions in Electronic City.

## Intended Use

The dataset is provided to support reproducibility of the experiments reported in the associated research paper, including traffic prediction and congestion-state analysis.

## Citation

If you use this dataset, please cite the associated research paper and this repository.

## License

This dataset is provided for academic and research use.