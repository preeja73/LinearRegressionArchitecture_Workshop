# Linear Regression Architecture Workshop

This project implements univariate linear regression on housing data, demonstrating both sklearn-based and from-scratch implementations using gradient descent, following Robot PM MLOps design principles.

---

## Summary of Work Completed

### 1. Data Sources Implementation
- **CSV Data**: Loaded California housing dataset from scikit-learn and saved to CSV format
- **API Data**: Integrated Toronto Open Data API to fetch data programmatically
- **Database Storage**: Implemented SQLite database storage for structured data management
- **Processed Data**: Cleaned and processed data stored in `data/processed/california_clean.csv` for model training

---

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries using `describe()`
- Data visualization with histograms and scatter plots
- Data quality assessment

---

### 3. Linear Regression Models
- **Model 1 (sklearn)**: Standard Linear Regression using scikit-learn
- **Model 2 (From Scratch)**: Custom implementation using gradient descent
- Both models trained on the same dataset
- Evaluation using RMSE, MAE, and R²

---

### 4. Modular Architecture
- **Data Loading**: `src/data_loader.py`
- **Preprocessing**: `src/preprocessing.py`
- **Model Training**: `src/model.py`
- **Evaluation**: `src/evaluation.py`
- **Pipeline**: `src/run_experiment.py`

---

### 5. Configuration Management
- YAML-based configuration file: `configs/experiment_config.yaml`
- Centralized parameter management for reproducibility

---

### 6. Experiment Tracking
- Results stored in `experiments/results.csv`
- Metrics tracked for each run

---

## Key Design Decisions

### 1. Modular Code Structure
Separated functionality into modules (`data_loader`, `preprocessing`, `model`, `evaluation`) to improve:
- Reusability
- Maintainability
- Testability
- Extensibility

---

### 2. Dual Model Implementation
Implemented both sklearn and from-scratch models to:
- Demonstrate algorithm understanding
- Validate results
- Enable comparison

---

### 3. Multiple Data Source Support
CSV, API, and database sources were implemented to simulate real-world ML pipelines.

---

### 4. Configuration-Driven Approach
YAML configuration allows:
- Easy experiment reproduction
- Parameter tuning without code changes
- Version control of experiment setups

---

### 5. Structured Project Layout
Project uses separate folders for:
- Raw data
- Processed data
- Notebooks
- Source code
- Configurations
- Experiments

This improves clarity and scalability.

---

### 6. Evaluation Metrics
- RMSE: penalizes large errors
- MAE: intuitive error metric
- R²: variance explained

---

### 7. Notebook-Based Workflow
- EDA and learning implemented in notebooks
- Architecture documented in `RobotPM_MLOps.ipynb`

---

## Project Structure

LinearRegressionArchitecture_Workshop/
├── configs/
│ └── experiment_config.yaml
├── data/
│ ├── raw/
│ │ ├── california.csv
│ │ ├── housing.sqlite
│ │ └── toronto_api_raw.json
│ └── processed/
│ ├── california_clean.csv
│ └── california_from_sql.csv
├── experiments/
│ ├── plots/
│ └── results.csv
├── notebooks/
│ ├── EDA.ipynb
│ ├── linear_regression.ipynb
│ └── RobotPM_MLOps.ipynb
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── evaluation.py
│ └── run_experiment.py
├── requirements.txt
└── README.md

Dependencies

pandas

numpy

matplotlib

scikit-learn

sqlalchemy

requests

pyyaml