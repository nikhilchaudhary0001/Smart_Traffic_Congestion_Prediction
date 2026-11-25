# 🚦 Smart Traffic Congestion Prediction (TRAFFICIQ)

An intelligent machine-learning-driven traffic congestion prediction system and command center designed to forecast next-hour road probe counts and traffic density using historical observations, spatio-temporal dynamics, and road network characteristics.

---

## 📌 Overview

Urban traffic congestion leads to billions of dollars in economic loss, increased emissions, and travel delays. **Smart Traffic Congestion Prediction** applies supervised machine learning to historical probe observations to anticipate traffic surges before they occur.

The project features:
- **Jupyter Notebook Pipeline**: Data inspection, exploratory data analysis (EDA), feature engineering, model benchmarking (Linear Regression, Polynomial Regression, Random Forest), and evaluation.
- **TRAFFICIQ Command Center (`trafficiq_app.py`)**: A modern, responsive Streamlit dashboard featuring live road-network filters, interactive route maps, rush-hour alerts, feature importance visualization, and scenario simulation.
- **Prototype Streamlit App (`app.py`)**: A focused prediction utility for quick inference across road segments.
- **Data Preparation Script (`prepare_trafficiq_data.py`)**: Pipeline for extracting and structuring road geometry and temporal observations.

---

## 🏗️ Architecture & Features

### 1. Feature Engineering (17 Spatio-Temporal Features)
- **Lagged Observations**: `probeCount_lag_1`, `probeCount_lag_2`, `probeCount_lag_3`, `probeCount_lag_24` (previous hour, 2h, 3h, and day-prior).
- **Rolling Statistics**: `probeCount_rolling_mean_3h`, `probeCount_rolling_std_3h` to capture short-term velocity and volatility.
- **Temporal Cyclical Encoding**: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos` to preserve circular time relationships.
- **Calendar Signals**: `hour`, `day_of_week`, `is_weekend`.
- **Road Network Attributes**: `speedLimit`, `frc` (Functional Road Class: Highway, Arterial, Collector, Local), and segment `distance`.

### 2. Model Performance Benchmarks

| Model | MAE (Probe Count) | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 24.95 | 46.84 | 0.9475 |
| **Polynomial Regression** | 22.59 | 43.10 | 0.9556 |
| **Random Forest Regression** ⭐ | **19.78** | **39.17** | **0.9633** |

*Trained on 500,000+ spatial observations from New Delhi road network dataset.*

---

## 📁 Repository Structure

```plaintext
Smart_Traffic_Congestion_Prediction/
├── Smart_Traffic_Congestion_Prediction.ipynb   # End-to-end data science notebook
├── trafficiq_app.py                           # Full-featured TRAFFICIQ command center
├── app.py                                     # Core prototype Streamlit interface
├── app_backup.py                              # Application backup
├── prepare_trafficiq_data.py                  # Data processing & aggregation script
├── 01_inspect_dataset.py                      # Dataset integrity & structure audit
├── 02_inspect_data_structures.py              # Schema & spatial feature audit
├── requirements.txt                           # Project dependencies
├── .gitignore                                 # Git ignore configuration
├── saved_model/
│   ├── model_features.json                    # 17 engineered feature names
│   ├── model_metrics.json                     # Evaluated benchmark metrics
│   └── random_forest_model.joblib             # Trained model weights (see note below)
└── trafficiq_data/
    ├── roads.json                             # Road segments with coordinates & attributes
    └── observations.csv.gz                    # Compressed historical probe time-series
```

> **Note on Model Weights (`saved_model/random_forest_model.joblib`)**:
> Due to GitHub's 100MB file limit, the 789 MB trained Random Forest `.joblib` model binary is excluded from the Git repository.
> - The `trafficiq_app.py` command center includes a high-fidelity intelligence estimation mode that runs immediately with the included `trafficiq_data/` files even without the `.joblib` file.
> - To regenerate the full 789MB model binary locally, execute `Smart_Traffic_Congestion_Prediction.ipynb`.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/nikhilchaudhary0001/Smart_Traffic_Congestion_Prediction.git
cd Smart_Traffic_Congestion_Prediction
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch TRAFFICIQ Command Center
```bash
streamlit run trafficiq_app.py
```
Or launch the prototype prediction app:
```bash
streamlit run app.py
```

---

## 📊 Dataset Information

The data contains urban probe counts collected across New Delhi's road infrastructure:
- **Spatial Coverage**: Highways, primary arterials, secondary roads, and local streets.
- **Probe Observations**: Aggregated vehicle probe readings indexed across hourly intervals.
- **Attributes**: Road geometry (start/end latitude and longitude), speed limits, functional classification (`frc`), and segment distance.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: `scikit-learn`, `joblib`
- **Data Engineering**: `pandas`, `numpy`, `pyarrow`
- **Visualization & UI**: `streamlit`, `plotly`, `pydeck`, `matplotlib`, `seaborn`
- **Notebook**: `jupyterlab` / `notebook`

---

## 👤 Author

- **GitHub**: [@nikhilchaudhary0001](https://github.com/nikhilchaudhary0001)
