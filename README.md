# Linear Regression Architecture Workshop

This project implements univariate linear regression on housing data, demonstrating both sklearn-based and from-scratch implementations using gradient descent.

## Summary of Work Completed

### 1. Data Sources Implementation
- **CSV Data**: Loaded California housing dataset from scikit-learn and saved to CSV format
- **API Data**: Integrated Toronto Open Data API to fetch data programmatically
- **Database Storage**: Implemented SQLite database storage for structured data management
- **Processed Data**: Cleaned and processed data stored in `data/processed/california_clean.csv` for model training

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries using `describe()`
- Data visualization with histograms
- Data quality assessment

### 3. Linear Regression Models
- **Model 1 (sklearn)**: Standard Linear Regression using scikit-learn
- **Model 2 (From Scratch)**: Custom implementation using gradient descent algorithm
- Both models trained on the same dataset with identical train-test splits
- Comprehensive evaluation using RMSE, MAE, and R² metrics

### 4. Modular Architecture
- **Data Loading**: `src/data_loader.py` - CSV file loading functionality
- **Preprocessing**: `src/preprocessing.py` - Train-test split utilities
- **Model Training**: `src/model.py` - Model training functions
- **Evaluation**: `src/evaluation.py` - Performance metric calculations

### 5. Configuration Management
- YAML-based configuration file (`configs/experiment_config.yaml`)
- Centralized parameter management for reproducibility

### 6. Experiment Tracking
- Results storage in CSV format (`experiments/results.csv`)
- Model comparison framework

## Key Design Decisions

### 1. **Modular Code Structure**
- **Decision**: Separated functionality into distinct modules (`data_loader`, `preprocessing`, `model`, `evaluation`)
- **Rationale**: 
  - Promotes code reusability and maintainability
  - Enables easy testing and debugging
  - Follows single responsibility principle
  - Facilitates collaboration and future extensions

### 2. **Dual Model Implementation**
- **Decision**: Implemented both sklearn and from-scratch gradient descent models
- **Rationale**:
  - Educational value: demonstrates understanding of underlying algorithms
  - Performance comparison: validates custom implementation against industry standard
  - Flexibility: from-scratch allows customization and deeper control

### 3. **Multiple Data Source Support**
- **Decision**: Implemented CSV, API, and database data sources
- **Rationale**:
  - Real-world applicability: different projects use different data sources
  - Demonstrates versatility in data handling
  - Prepares for production scenarios with diverse data pipelines

### 4. **Configuration-Driven Approach**
- **Decision**: Used YAML configuration file for experiment parameters
- **Rationale**:
  - Reproducibility: easy to track and reproduce experiments
  - Flexibility: change parameters without modifying code
  - Best practice: aligns with MLOps principles
  - Version control friendly: configurations can be tracked in git

### 5. **Structured Project Layout**
- **Decision**: Organized project with separate folders for data (raw and processed), notebooks, source code, configs, and experiments
- **Rationale**:
  - Clear separation of concerns: raw data vs processed data
  - Scalability: easy to add new components
  - Industry standard: follows common ML project structures
  - Maintainability: easier navigation and organization
  - Data pipeline clarity: distinguishes between raw input and cleaned data ready for modeling

### 6. **Comprehensive Evaluation Metrics**
- **Decision**: Used RMSE, MAE, and R² for model evaluation
- **Rationale**:
  - RMSE: Penalizes large errors, same units as target variable
  - MAE: Provides intuitive error interpretation
  - R²: Explains variance explained by the model
  - Multiple metrics provide holistic performance assessment

### 7. **Notebook-Based Workflow**
- **Decision**: Used Jupyter notebooks for exploration and analysis
- **Rationale**:
  - Interactive development and visualization
  - Educational tool: step-by-step execution
  - Documentation: code and results in one place
  - Easy sharing and presentation

## Project Structure

```
LinearRegressionArchitecture_Workshop/
├── data/
│   ├── raw/              # Raw data files (CSV, database)
│   │   ├── california.csv
│   │   └── housing.db
│   └── processed/        # Processed/cleaned data files
│       └── california_clean.csv
├── notebooks/            # Jupyter notebooks for analysis
│   ├── EDA.ipynb         # Exploratory Data Analysis
│   └── linear_regression.ipynb  # Main regression implementation
├── src/                  # Modular source code
│   ├── data_loader.py    # Data loading utilities
│   ├── preprocessing.py  # Data preprocessing functions
│   ├── model.py          # Model training functions
│   └── evaluation.py     # Evaluation metrics
├── configs/              # Configuration files
│   └── experiment_config.yaml
├── experiments/          # Experiment results
│   └── results.csv
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## How to Run

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks**
   - Open Jupyter: `jupyter notebook` or `jupyter lab`
   - Navigate to `notebooks/` folder
   - Run `EDA.ipynb` for data exploration
   - Run `linear_regression.ipynb` for model implementation and comparison

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Data visualization
- scikit-learn: Machine learning algorithms and utilities
- sqlalchemy: Database operations
- requests: API data fetching
- pyyaml: Configuration file parsing

## Results

The project compares two linear regression implementations:
- **Sklearn Model**: Industry-standard implementation
- **From Scratch Model**: Custom gradient descent implementation

Both models are evaluated on the same test set using RMSE, MAE, and R² metrics, with results stored in `experiments/results.csv`.
