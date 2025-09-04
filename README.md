# NYC Taxi Trip Duration Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange?logo=scikit-learn" alt="Scikit-Learn Version">
  <img src="https://img.shields.io/badge/Pandas-1.5%2B-blue?logo=pandas" alt="Pandas Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

A comprehensive machine learning solution for predicting taxi trip durations in New York City using spatial, temporal, and trip metadata features. This enterprise-grade project demonstrates a complete data science pipeline from exploratory analysis to production-ready model deployment.

![NYC Taxi](asset/graphs/Dropoff%20Location%20Density%20Heatmap.png)

## Project Overview

This project tackles the challenge of predicting taxi trip durations in New York City using machine learning techniques. The solution incorporates comprehensive data processing, advanced feature engineering, and optimized model training to deliver accurate predictions that could benefit taxi companies, ride-sharing platforms, and urban transportation planners.

The final Ridge Regression model achieves a validation R² score of 0.6411, demonstrating strong predictive performance. The project follows software engineering best practices with a modular structure, thorough documentation, and reproducible workflows.

## Tech Stack & Libraries

<div align="center">

| Category | Technologies |
| :--- | :--- |
| **Core Programming** | <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python"> |
| **Data Processing** | <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas"> <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy"> |
| **Machine Learning** | <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn"> <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" alt="SciPy"> |
| **Data Visualization** | <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib"> <img src="https://img.shields.io/badge/Seaborn-4B77B0?logo=seaborn&logoColor=white" alt="Seaborn"> |
| **Notebook Environment** | <img src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white" alt="Jupyter"> |

</div>

## Dataset Description

The dataset contains historical NYC taxi trip records with the following key features:
- Pickup and dropoff coordinates (longitude, latitude)
- Pickup datetime with temporal components
- Passenger count
- Vendor information
- Target variable: trip duration in seconds

## Exploratory Data Analysis

### Data Distribution and Outliers

![Trip Duration Distribution](asset/graphs/Original%20Trip%20Duration%20Distribution.png)
*Original distribution of trip durations showing right-skewed pattern with extreme outliers*

![Log-Transformed Distribution](asset/graphs/Log-Transformed%20Trip%20Duration%20Distribution.png)
*Log-transformed distribution demonstrating improved normality for modeling*

### Temporal Patterns

![Hourly Effect](asset/graphs/Hour%20of%20Day%20Effect%20on%20Trip%20Duration.png)
*Trip duration variation by hour of day showing peak hours with longer durations*

![Weekly Pattern](asset/graphs/Day%20of%20Week%20Effect%20on%20Trip%20Duration.png)
*Weekly patterns in trip duration with noticeable weekday/weekend variations*

### Spatial Analysis

![Dropoff Heatmap](asset/graphs/Dropoff%20Location%20Density%20Heatmap.png)
*Density heatmap of dropoff locations highlighting popular destinations across NYC*

![Spatial Features](asset/graphs/Feature%20vs%20Feature%20pickup_longitude%20vs%20dropoff_longitude.png)
*Relationship between pickup and dropoff longitude values*

### Feature Relationships

![Correlation Heatmap](asset/graphs/Correlation%20Heatmap.png)
*Correlation matrix showing relationships between numerical features*

![Distance vs Duration](asset/graphs/Distance%20vs%20Trip%20Duration.png)
*Relationship between trip distance and duration with clear positive correlation*

![Pairplot](asset/graphs/Pairplot%20of%20Numerical%20Features%20.png)
*Pairwise relationships between numerical features*

### Data Quality Assessment

![Missing Values](asset/graphs/isna.png)
*Missing value analysis showing data completeness across features*

![Numerical Distributions](asset/graphs/plot_numerical_distributions.png)
*Distribution analysis of numerical features*

## Feature Engineering

The project implements sophisticated feature engineering techniques:

- **Spatial features**: Haversine distance, Manhattan distance, bearing direction
- **Temporal features**: Hour of day, day of week, month, is_weekend flag
- **Spatial clustering**: Identification of high-density pickup/dropoff zones
- **Statistical features**: Rolling averages, trip frequency by area/time
- **Interaction terms**: Cross-features between spatial and temporal dimensions

## Model Training & Pipeline

The modeling approach utilizes a Ridge Regression pipeline with careful feature selection and hyperparameter optimization:

- **Preprocessing**: Standard scaling, outlier handling, missing value imputation
- **Feature selection**: Correlation analysis, recursive feature elimination
- **Model tuning**: Grid search with cross-validation for optimal regularization
- **Validation**: Stratified temporal cross-validation to prevent data leakage
- **Interpretation**: SHAP analysis for model explainability

## Results & Performance

The final model demonstrates strong predictive performance:

| Metric | Value |
|--------|-------|
| Cross-Validation Mean R² | 0.6089 |
| Validation R² | 0.6411 |
| Validation RMSE | 0.4924 |
| Validation RMSLE | 0.0732 |

**Cross-Validation Performance by Fold**:
- Fold 1: R² = 0.6185
- Fold 2: R² = 0.6207
- Fold 3: R² = 0.5905
- Fold 4: R² = 0.6182
- Fold 5: R² = 0.5963

The model shows consistent performance across validation folds with no significant overfitting.

## Project Structure

<details>
<summary>Click to expand project directory structure</summary>

```
nyc-taxi-trip-duration/
├── .env.example                 # Environment variables template
├── .gitattributes              # Git configuration
├── .gitignore                  # Git ignore rules
├── LICENSE                     # Project license
├── Notebook/
│   └── EDA_TripDuration.ipynb  # Comprehensive exploratory data analysis
├── asset/
│   └── graphs/                 # Visualization assets
│       ├── Average Trip Duration.png
│       ├── Boxplot: {col} vs trip_duration .png
│       ├── Correlation Heatmap.png
│       ├── Day of Week Effect on Trip Duration.png
│       ├── Distance vs Trip Duration.png
│       ├── Dropoff Location .png
│       ├── Dropoff Location Density Heatmap.png
│       ├── Feature vs Feature: pickup_longitude vs dropoff_longitude.png
│       ├── Hour of Day Effect on Trip Duration.png
│       ├── Log-Transformed Trip Duration Distribution.png
│       ├── Original Trip Duration Distribution.png
│       ├── Pairplot of Numerical Features .png
│       ├── Trip Duration over Time.png
│       ├── Trip Duration.png
│       ├── isna.png
│       └── plot_numerical_distributions.png
├── data/                       # Raw and processed data
│   └── data_split0/
│       ├── create_metadata_zip.py
│       ├── metadata.json
│       ├── test.zip
│       ├── train.zip
│       └── val.zip
├── data_processed/             # Processed data assets
│   ├── metadata.json
│   └── train.zip
├── models/                     # Trained model artifacts
│   └── ridge_pipeline.pkl      # Production model
├── requirements.txt            # Python dependencies
├── results/                    # Analysis outputs
│   ├── cv_scores.csv
│   ├── data_processed.zip
│   ├── predictions.csv
│   ├── selected_features.csv
│   ├── shap_feature_importance.csv
│   ├── test_with_predictions.csv
│   ├── top10_shap_features.csv
│   ├── validation_metrics.csv
│   └── validation_predictions.csv
├── src/                        # Source code
│   ├── cross_validation.py     # Cross-validation implementation
│   ├── evaluate.py             # Model evaluation utilities
│   ├── feature_engineering.py  # Feature engineering pipeline
│   ├── feature_selection.py    # Feature selection methods
│   ├── main.py                 # Main execution script
│   ├── modeltraining.py        # Model training workflow
│   └── pipeline.py             # ML pipeline definition
└── submissions/                # Prediction outputs
    └── test_predictions.csv
```
</details>

## Installation & Usage

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nyc-taxi-trip-duration.git
   cd nyc-taxi-trip-duration
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the complete pipeline**
   ```bash
   python src/main.py
   ```

### Advanced Usage

**Execute individual components**:
```bash
# Feature engineering
python src/feature_engineering.py

# Model training
python src/modeltraining.py

# Cross-validation
python src/cross_validation.py

# Model evaluation
python src/evaluate.py
```

**Explore the Jupyter notebook**:
```bash
jupyter notebook Notebook/EDA_TripDuration.ipynb
```

## Future Work

Potential enhancements for the project:

1. **Model Improvements**
   - Experiment with gradient boosting models (XGBoost, LightGBM)
   - Implement ensemble approaches
   - Add deep learning architectures for spatial-temporal patterns

2. **Feature Engineering**
   - Incorporate weather data
   - Add real-time traffic information
   - Include points of interest data

3. **Deployment**
   - Create API endpoints for predictions
   - Develop real-time prediction capabilities
   - Implement model monitoring and retraining pipelines

4. **Visualization**
   - Interactive dashboard for exploration
   - Real-time prediction visualization
   - Model performance monitoring tools

## Acknowledgments

- New York City Taxi and Limousine Commission for providing the dataset
- Open-source community for the Python data science ecosystem
- Contributors to the scikit-learn, pandas, and matplotlib libraries

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or contributions, please open an issue or submit a pull request.
