# Insurance Claims Prediction Pipeline

A production-ready machine learning pipeline for predicting insurance claim amounts using MLflow for experiment tracking, Vetiver for model deployment, and Pins for model versioning. This project provides script-based workflow execution options.

## Project Overview

This project predicts the `Ultimate_Claim_Amount` for insurance claims using various machine learning models. It includes comprehensive data processing, feature engineering, model training, and deployment capabilities.

## Features

- **Data Processing Pipeline**: Automated data loading, validation, and feature engineering
- **Feature Engineering**: 10+ derived features including risk indicators and temporal features
- **Multi-Model Training**: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- **Hyperparameter Tuning**: Automated grid search with cross-validation
- **MLflow Integration**: Complete experiment tracking and model registry
- **Vetiver**: Standardized model deployment format
- **Pins**: Model versioning and storage
- **Data Validation**: Comprehensive quality checks

## Project Structure

```
insurance-claims-prediction/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── model_params.yaml        # Model hyperparameters
│   └── feature_config.yaml      # Feature engineering config
├── data/
│   ├── raw/                     # Raw input data
│   ├── interim/                 # Intermediate data
│   └── processed/               # Processed data
├── src/                         # Source code modules
├── scripts/                     # Individual scripts (traditional approach)
├── models/                      # Saved models and preprocessors
├── model_pins_board/            # Pins model versions
├── mlruns/                      # MLflow experiment tracking
├── logs/                        # Log files
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd insurance-claims-prediction
```

2. **Create virtual environment**:
```bash
python -m venv adm_venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```
4. **Create necessary directories**:
```bash
mkdir -p data/raw data/interim data/processed logs mlruns model_pins_board models results
```

6. **Add your data**:
Place your CSV files in the `data/raw/` directory:
- `claims_table.csv`
- `policyholder_table.csv`

## Usage

### Option 1: Traditional Script-Based Approach

#### Step 1: Data Preparation
Run the data preparation pipeline to load, validate, and engineer features:

```bash
python scripts/prepare_data.py
```

This script will:
- Load and merge claims and policy data
- Validate data quality
- Handle missing values
- Create derived features
- Save processed data

#### Step 2: Train Models
Run the model training pipeline:

```bash
python scripts/create_insurance_model.py
```

This script will:
- Load processed data
- Split into train/test sets
- Apply preprocessing transformations
- Train all enabled models with hyperparameter tuning
- Log results to MLflow
- Version models with Pins
- Compare model performance

#### Step 3: View Results
Start the MLflow UI to view experiment results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Pipeline Stages

### Data Loading
- Loads claims and policy CSV files
- Merges datasets on Policy_ID and Customer_ID
- **Output**: `data/interim/merged_raw.csv`

### Data Validation
- Validates data quality and completeness
- Checks for missing values and duplicates
- **Output**: `data/processed/data_validation_report.txt`

### Feature Engineering
- Creates derived features (FNOL_Delay, Driver_Age, etc.)
- Handles missing values
- **Output**: `data/processed/processed_claims.csv`

### Preprocessing
- Creates preprocessing pipeline (StandardScaler + OneHotEncoder)
- Splits data into train/test sets
- **Outputs**:
  - `models/preprocessor.joblib`
  - `data/interim/train_test_data_split.joblib`

### Parallel Model Training
- Trains 4 models simultaneously:
  1. Linear Regression
  2. Random Forest (with hyperparameter tuning)
  3. Gradient Boosting (with hyperparameter tuning)
  4. XGBoost (with hyperparameter tuning)
- **Outputs per model**:
  - `results/models/{model}/metrics.json`
  - `results/models/{model}/model.pkl`
  - MLflow tracking data
  - Pinned model versions

### Results Aggregation
- Compares all models
- Generates comparison report
- Selects best model
- **Outputs**:
  - `results/model_comparison.csv`
  - `results/model_comparison.html`
  - `results/best_model_name.txt`
  - `results/pipeline_summary.txt`

## Feature Engineering

The pipeline creates the following derived features:

### Temporal Features
- **FNOL_Delay**: Days between accident and FNOL reporting
- **Settlement_Delay**: Days between FNOL and settlement

### Demographic Features
- **Driver_Age**: Driver age at time of accident (years)
- **License_Age**: Years since license issue
- **Vehicle_Age**: Vehicle age at time of accident (years)

### Risk Indicators
- **High_Risk_Driver**: Flag for drivers < 25 or > 70 years old
- **Inexperienced_Driver**: Flag for drivers with < 2 years license
- **Old_Vehicle**: Flag for vehicles > 10 years old
- **Early_FNOL**: Flag for FNOL reported within 1 day

## Configuration

### Main Configuration (`config/config.yaml`)

```yaml
project:
  name: "insurance-claims-prediction"
  version: "0.1.0"

data:
  claims_file: "data/raw/claims_table.csv"
  policy_file: "data/raw/policyholder_table.csv"
  test_size: 0.2
  random_state: 42

models:
  target_variable: "Ultimate_Claim_Amount"
  enabled:
    - linear_regression
    - random_forest
    - gradient_boosting
    - xgboost
```

### Model Parameters (`config/model_params.yaml`)
Customize hyperparameter search grids for each model.

### Feature Configuration (`config/feature_config.yaml`)
Define feature engineering rules, imputation strategies, and feature lists.

## Models

### Available Models

1. **Linear Regression**: Baseline model
2. **Random Forest**: Ensemble method with bagging
3. **Gradient Boosting**: Sequential ensemble method
4. **XGBoost**: Optimized gradient boosting

### Evaluation Metrics

All models are evaluated using:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: R-squared score

## Data Requirements

### Claims Table (`claims_table.csv`)
Required columns:
- Policy_ID
- Customer_ID
- Claim_ID
- Accident_Date
- FNOL_Date
- Settlement_Date
- Estimated_Claim_Amount
- Ultimate_Claim_Amount
- Traffic_Condition
- Weather_Condition

### Policyholder Table (`policyholder_table.csv`)
Required columns:
- Policy_ID
- Customer_ID
- Date_of_Birth
- Full_License_Issue_Date
- Vehicle_Year
- (Other relevant policy details)

## Loading Trained Models

### From Pins Board

```python
from src.pins_utils.board_manager import PinsBoardManager
from src.utils.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader().get_main_config()

# Initialize pins manager
pins_manager = PinsBoardManager(config)

# List available models
models = pins_manager.list_models()
print(models)

# Load a specific model
model = pins_manager.load_model("xgboost")

# Load preprocessor
import joblib
preprocessor = joblib.load("models/preprocessor.joblib")

# Make predictions
X_new_processed = preprocessor.transform(X_new)
predictions = model.model.predict(X_new_processed)
```

### From MLflow

```python
import mlflow

# Load model by run ID
model_uri = f"runs:/<run_id>/model"
model = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = model.predict(X_new_processed)
```

## Advantages of Snakemake Workflow

### 1. Automatic Dependency Tracking
**Before (Scripts)**:
```bash
python scripts/prepare_data.py
python scripts/train_insurance_models.py  # Must remember to run after data prep
```

**With Snakemake**:
```bash
snakemake --cores 4  # Automatically knows what to run and in what order
```

### 2. Incremental Builds
**Scenario**: You modify model hyperparameters

**Before (Scripts)**: Manually rerun training script

**With Snakemake**: 
```bash
# Edit config/model_params.yaml
snakemake --cores 4  # Only reruns training, skips data prep!
```

### 3. Parallel Execution
**Before**: Train models sequentially (2 hours total)
```
Linear Regression → Random Forest → Gradient Boosting → XGBoost
   (30 min)           (30 min)         (30 min)          (30 min)
```

## Workflow Diagram

```
data/raw/*.csv
      ↓
[load_raw_data]
      ↓
data/interim/merged_raw.csv
      ↓
[validate_data] → data_validation_report.txt
      ↓
[prepare_data]
      ↓
data/processed/processed_claims.csv
      ↓
[create_preprocessor]
      ↓
preprocessor.joblib + train/test splits
      ↓
      ├─→ [train_linear_regression] → results/models/linear_regression/*
      ├─→ [train_random_forest]     → results/models/random_forest/*
      ├─→ [train_gradient_boosting] → results/models/gradient_boosting/*
      └─→ [train_xgboost]           → results/models/xgboost/*
                ↓
         [aggregate_metrics]
                ↓
    model_comparison.csv/html
                ↓
       [select_best_model]
                ↓
      best_model_name.txt
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` when running training script
```
Solution: Run prepare_data.py first to create processed data
```

**Issue**: Missing columns in data
```
Solution: Ensure your CSV files contain all required columns
```

**Issue**: Memory error during training
```
Solution: Reduce parameter grid size in config/model_params.yaml
```

**Issue**: Snakemake "MissingInputException"
```
Solution: Check that raw data files are in `data/raw/`
```

**Issue**: Snakemake "Rule execution failed"
```
Solution: Check logs in `logs/snakemake/`
```

**Issue**: Snakemake "Nothing to be done"
```
Solution: All outputs are up-to-date. Use --forceall to rerun everything
```

## Best Practices

1. **Always run data preparation first** when using scripts: `prepare_data.py` before `train_insurance_models.py`
2. **Use Snakemake for production workflows** for better dependency management
3. **Check data validation results**: Review logs for data quality issues
4. **Monitor MLflow experiments**: Track performance across runs
5. **Version your configurations**: Keep track of config changes
6. **Save preprocessor**: Always save the fitted preprocessor for deployment
7. **Use `snakemake -n` first** to see what will run
8. **Check logs** in `logs/snakemake/` if errors occur
9. **Use `--cores` wisely** - don't exceed your CPU count

## Performance Tips

### Optimize for Your Hardware
```bash
# For laptop (2-4 cores)
snakemake --cores 2

# For workstation (8+ cores)
snakemake --cores 8

# For server (16+ cores)
snakemake --cores 16
```

### Reduce Hyperparameter Grid
Edit `config/model_params.yaml`:
```yaml
random_forest:
  param_grid:
    n_estimators: [100]  # Instead of [50, 100, 200]
    max_depth: [20]      # Instead of [10, 20, null]
```

### Cache Preprocessing
Preprocessing runs once and results are cached. All models use the same preprocessed data.

## Extending the Workflow

### Add a New Model
1. Create model class in `src/models/new_model.py`
2. Add to `MODELS` list in Snakefile
3. Add training rule similar to existing models
4. Run: `snakemake --cores 4`

### Add New Features
1. Edit `config/feature_config.yaml`
2. Run: `snakemake --cores 4`
3. Only feature engineering and downstream steps rerun!

### Add Validation Step
```python
rule validate_models:
    input:
        expand("results/models/{model}/model.pkl", model=MODELS)
    output:
        "results/validation_report.txt"
    shell:
        "python scripts/validate_all_models.py"
```

## Integration with Existing Tools

### MLflow
All model training logs to MLflow automatically. View with:
```bash
mlflow ui
```

### Pins
Models are versioned in `model_pins_board/`. Load with:
```python
from src.pins_utils.board_manager import PinsBoardManager
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().get_main_config()
pins_manager = PinsBoardManager(config)
model = pins_manager.load_model("xgboost")
```

### Jupyter Notebooks
Use processed data in notebooks:
```python
import pandas as pd
df = pd.read_csv("data/processed/processed_claims.csv")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

## Support

For issues and questions:
- Check the logs in `logs/` directory
- Review MLflow UI for experiment details
- For Snakemake issues, check `logs/snakemake/`
- Open an issue on GitHub

## Authors

Victor Oseji

## Acknowledgments

- MLflow for experiment tracking
- Vetiver for model deployment
- Pins for model versioning
- Snakemake for workflow management
- scikit-learn for machine learning algorithms