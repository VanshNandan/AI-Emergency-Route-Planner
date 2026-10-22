
# AI Emergency Route Planner

An AI-powered route-planning project that uses predicted
road congestion to identify efficient routes for emergency
travel.

## Overview

Emergency vehicles need efficient routes to reach their
destinations. This project combines machine learning-based
congestion prediction with graph-based route planning to
estimate travel time and identify suitable paths across a
road network.

The project includes a Streamlit interface for interacting
with route-planning results.

## Features

- Traffic congestion prediction using machine learning
- Decision Tree and Random Forest model comparison
- Graph-based route planning
- Estimated travel time for selected routes
- Streamlit interface for displaying results
- Model evaluation using MAE, RMSE, and R-squared

## Technology Stack

- Python
- Scikit-learn
- Streamlit
- NumPy and Pandas (if used in the final application)
- Graph-based route search

## Dataset

The project uses the Electronic City Traffic Dataset
for Bengaluru, India.

The dataset contains synthetic traffic time-series data,
including vehicle speed, vehicle count, and congestion
information.

- 17,280 timestamps
- 602 directed road segments
- 262 road-network nodes
- Five-minute temporal resolution

The dataset is synthetic and should not be interpreted
as live traffic sensor data.

## Machine Learning Results

The Random Forest model performed better than the
Decision Tree model on the evaluated test dataset.

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Decision Tree | 0.0725 | 0.0973 | 0.7967 |
| Random Forest | 0.0683 | 0.0900 | 0.8258 |

These metrics evaluate congestion prediction, not
real-world emergency response performance.

## Project Structure

Adapt this outline to match the actual repository.

- `README.md` - Project documentation
- `train_models.py` - Model training script, if applicable
- `models/` - Model artifacts, if distributed separately
- `screenshots/` - Application screenshots
- Streamlit entry-point script - Application interface

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies.
4. Prepare the dataset and model artifacts as documented.
5. Run the training script if model artifacts are not
   provided.
6. Start the Streamlit application.

Example commands:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model, if required:

```bash
python train_models.py
```

Run the application, replacing `app.py` with the
actual Streamlit entry-point filename:

```bash
streamlit run app.py
```

## Screenshots

Add screenshots of the running application to the
`screenshots/` directory and link them here.

## Limitations

- The traffic dataset is synthetic.
- Predictions depend on the training data and model.
- Estimated travel times are experimental and are not
  validated emergency-response guarantees.
- The application is a project prototype, not a
  production emergency dispatch system.

## Future Improvements

- Integration with validated, real-time traffic sources
- Improved route optimization
- Additional model evaluation and validation
- Deployment as an accessible web application

## Author

Vansh Nandan  
B.Tech. Computer and Communication Engineering  
Manipal University Jaipur
