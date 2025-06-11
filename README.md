# FGRPre
Machine Learning-Enhanced Prediction of Fetal Growth Restriction via Cardiac Remodeling Phenotypes

## Machine Learning-Enhanced Prediction of Fetal Growth Restriction via Cardiac Remodeling Phenotypes
## Overview
FGRPre is a collection of machine learning models for predicting fetal growth restriction (FGR) using cardiac remodeling phenotypes. The repository contains scripts to train, evaluate and visualize these models.

## Repository Structure
- `train.py` – training pipeline with multiple model implementations
- `test.py` – evaluate trained models on an Excel dataset
- `visual.py` – SHAP-based model interpretation utilities
- `utils/` – common helpers for metrics and data preprocessing
- `results/` – output directory (create before running code)
- `visual/` – folder for generated visualizations
- `README.md` – project documentation

## Requirements
Python 3.8 or later and the following packages are required:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

## Data Preparation
The dataset is not included in this repository. Place your Excel file under `dataset/` and update the path in the scripts if needed. An example path is `dataset/Final_data_zscore.xlsx`.

## Quick Start
1. **Train models**
   ```bash
   python train.py
   ```
   Trained models are saved under `train_models/`.
2. **Evaluate models**
   ```bash
   python test.py
   ```
   Results such as ROC curves and confusion matrices are written to `results/`.
3. **Visualize models**
   ```bash
   python visual.py
   ```
   Generates SHAP visualizations for model interpretation.

## Further Reading
- Study `train.py` and `test.py` to understand data flow and model definitions.
- See `utils/metric.py` for metric computation and plotting functions.
- Learn how SHAP explains model predictions and explore feature contributions.
- Feel free to experiment with additional models or feature engineering strategies.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
