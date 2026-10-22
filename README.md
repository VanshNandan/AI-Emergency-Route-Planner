# AI Emergency Route Planner 🚑

A machine learning-based route planning prototype that predicts road congestion and uses graph-based path planning to identify efficient routes for emergency travel.

## Overview

Emergency vehicles need efficient routes to reach their destinations. This project combines traffic congestion prediction with graph-based route planning to estimate route travel time and help identify suitable paths across a road network.

The application uses **Python, Scikit-learn, and Streamlit** to demonstrate how machine learning can support congestion-aware route planning.

## Features

- Predict road congestion using machine learning.
- Compare Decision Tree and Random Forest regression models.
- Plan routes over a directed road network.
- Estimate travel time for a selected route.
- Display route-planning results through a Streamlit interface.
- Evaluate model performance using MAE, RMSE, and R².

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy, Pandas
- **Interface:** Streamlit
- **Route Planning:** Graph-based path search

## Dataset

The project uses the **Electronic City Traffic Dataset** for Bengaluru, India.

The dataset contains synthetic traffic time-series data with vehicle speed, vehicle count, and congestion information.

| Dataset characteristic | Value |
|---|---:|
| Time-series timestamps | 17,280 |
| Directed road segments | 602 |
| Road-network nodes | 262 |
| Temporal resolution | 5 minutes |

**Important:** The dataset is synthetic, not live traffic sensor data. Results demonstrate a prototype and should not be interpreted as validated real-world emergency response performance.

## Model Performance

Two regression models were trained and evaluated for congestion prediction.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Decision Tree | 0.0725 | 0.0973 | 0.7967 |
| Random Forest | 0.0683 | 0.0900 | 0.8258 |

The Random Forest model achieved lower MAE and RMSE and a higher R² on the evaluated test dataset.

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

These metrics measure congestion prediction performance, not actual emergency response times.

## Project Structure

```text
AI-Emergency-Route-Planner/
├── data/
├── models/
├── Screenshots/
├── app.py
├── train_models.py
├── route_planner.py
├── inspect_dataset.py
├── inspect_network.py
├── check_matrix.py
├── plot_comparison.py
├── .gitignore
└── README.md
```

- `app.py` — Streamlit application.
- `train_models.py` — Trains and evaluates the machine learning models.
- `route_planner.py` — Route-planning logic.
- `inspect_dataset.py` — Inspects dataset information.
- `inspect_network.py` — Inspects the road network.
- `check_matrix.py` — Checks traffic matrix data.
- `plot_comparison.py` — Generates model comparison plots.
- `data/` — Dataset files.
- `models/` — Saved model artifacts and evaluation metrics.
- `Screenshots/` — Application screenshots, if added.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/VanshNandan/AI-Emergency-Route-Planner.git
cd AI-Emergency-Route-Planner
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows, activate it using:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the packages required by the project:

```bash
pip install numpy pandas scikit-learn streamlit matplotlib joblib
```

If the application reports any missing package, install the corresponding dependency as well.

### 4. Run the application

```bash
streamlit run app.py
```

### 5. Train the models (optional)

If you want to retrain the models, run:

```bash
python train_models.py
```

Retraining may overwrite saved model artifacts. Ensure the expected dataset files are present before running the training script.

## Screenshots

Application screenshots can be added here to demonstrate the Streamlit interface, model comparison, and route-planning results.

## Limitations

- The dataset is synthetic rather than live traffic data.
- Predictions depend on the dataset and trained model.
- Estimated travel times are experimental.
- The prototype is not a production emergency dispatch system and should not be used to make real emergency-routing decisions.

## Future Improvements

- Integrate validated real-time traffic data.
- Improve route optimization and travel-time estimation.
- Evaluate models using additional datasets and validation methods.
- Deploy the application for broader accessibility.
- Explore additional machine learning models.

## Author

**Vansh Nandan**  
B.Tech. — Computer and Communication Engineering  
Manipal University Jaipur

---

*Developed as an academic machine learning and route-planning project.*
