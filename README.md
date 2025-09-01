# NYC Taxi Trip Duration Prediction

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

 **Overview**  
This project develops machine learning models to predict New York City taxi trip durations using historical trip data. The implementation includes comprehensive data processing pipelines, feature engineering, and multiple machine learning approaches.

---

##  Project Structure


trip_duration_project/
├── data/

│ ├── raw/ # Original, immutable data

│ ├── processed/ # Processed data files

│ │ ├── data_processed.zip

│ │ └── metadata.json

│ └── splitted/ # Train/test/validation splits

├── models/ # Trained models

│ └── ridge_pipeline.pkl

├── notebooks/ # Jupyter notebooks

│ ├── EDA_TripDuration.ipynb

│ └── .ipynb_checkpoints/ # Notebook autosaves

├── reports/ # Analysis outputs and reports

│ ├── results/ # Experiment results

│ │ ├── cv_scores.csv

│ │ ├── feature_ranking.csv

│ │ ├── top_features.csv

│ │ ├── validation_predictions.csv

│ │ └── pipeline_selected_framework.csv

│ ├── graphs/ # Visualizations and plots

│ ├── asset/ # Additional assets

│ └── shap_feature_importance/ # SHAP analysis results

├── src/ # Source code

│ ├── pycache/

│ ├── cross_validation.py

│ ├── evaluate.py

│ ├── feature_engineering.py

│ ├── feature_selection.py

│ ├── main.py

│ ├── model_training.py

│ ├── pipeline.py

│ └── error_validation.csv

├── env/ # Environment configuration

│ └── env.sample

├── .gitattributes # Git attributes

├── .gitignore # Git ignore rules

├── LICENSE # MIT License

├── requirements.txt # Python dependencies

└── README.md # Project documentation




----

##  Installation

1. **Clone the repository**
```bash
git clone https://github.com/mohamedmahmod26/nyctaxi-trip-duration.git
cd nyctaxi-trip-duration
``` 

## Set up virtual environment
python -m venv venv
# Linux/MacOS
source venv/bin/activate
# Windows
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
--- 

# Run Complete Pipeline
python src/main.py
# Exploratory Data Analysis
jupyter notebook notebooks/EDA_TripDuration.ipynb
# Train Model
python src/model_training.py
# Cross-Validation
python src/cross_validation.py
# Evaluate Model
python src/evaluate.py
---
## Results

### Key Findings

- Top predictive features: reports/results/top_features.csv

- Feature importance rankings: reports/results/feature_ranking.csv

- Cross-validation performance: reports/results/cv_scores.csv

- Validation predictions: reports/results/validation_predictions.csv

# Model Performance

- The Ridge regression pipeline (models/ridge_pipeline.pkl) achieved the best performance with optimized hyperparameters.

## Contributing

1- Fork the project

2- Create your feature branch

git checkout -b feature/AmazingFeature


3- Commit your changes

git commit -m 'Add some AmazingFeature'


4- Push to the branch

git push origin feature/AmazingFeature


5- Open a Pull Request

## Acknowledgments

- NYC Taxi and Limousine Commission for providing the data

- Open-source community for the libraries and tools used in this project