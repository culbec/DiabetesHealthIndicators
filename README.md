# Diabetes Health Indicators

This repository contains experimental work on predicting diabetes-related outcomes using the
**Diabetes Health Indicators dataset (100k samples)**. The project focuses on implementing
machine learning models **from scratch**, validating them against optimized library
implementations, and analyzing their performance, stability, and interpretability.

The work was developed in the context of an academic machine learning project, with emphasis on:
- algorithmic correctness
- reproducible experimentation
- statistical evaluation
- model interpretability

---

## Project Scope

The repository addresses two supervised learning tasks:

### 1. Classification
- **Objective:** Predict diabetes diagnosis (binary classification)
- **Models:**
  - Random Forest from scratch
  - `sklearn.ensemble.RandomForestClassifier` (scikit-learn baseline)

### 2. Regression
- **Objective:** Predict a continuous diabetes risk score
- **Models:**
  - Support Vector Regression (SVR) from scratch
  - `sklearn.svm.SVR` (libsvm-based baseline)

Both tasks follow the same experimental protocol:
- statistical feature analysis and selection,
- cross-validation with hyperparameter optimization,
- held-out test evaluation,
- model comparison against library baselines.

---

## Repository Structure

```
├── dhi/                                # Core source package
│   ├── data/                           # Data handling
│   │   ├── loader/                     # Dataset loaders
│   │   ├── preprocessing/              # Feature scaling, encoding, splits
│   │   └── visualiation/               # Plotting
│   │
│   ├── models/                         # Machine learning core implementations
│   │   ├── eval/                       # Performance metrics implementation
│   │   │   ├── metrics.py
│   │   │   └── scorer.py
│   │   │
│   │   ├── experiments/                # Experiment pipelines
│   │   │   └── scorer.py
│   │   │
│   │   ├── inspection/                 # Model interpretability
│   │   │   └── shap_explainer.py
│   │   │
│   │   ├── random_forest/              # Random Forest (from scratch)
│   │   │   ├── tree/                   # Decision tree implementation
│   │   │   ├── sampling/               # Bagging implementation
│   │   │   └── forest/                 # Forest-level logic
│   │   │
│   │   ├── selection/                  # Model tuning
│   │   │   ├── grid_search.py
│   │   │   └── cross_validation.py
│   │   │
│   │   └── svr/                        # Support Vector Regression (from scratch)
│   │
│   ├── statistics/                     # Statistical analysis & feature selection
│   │   ├── feature_selection.py
│   │   ├── component_reduction.py
│   │   └── metrics.py
│   │
│   ├── constants.py                    # Project-wide constants
│   ├── utils.py                        # Logging, helpers, configuration utils
│   └── decorators.py                   # Timing, logging decorators
│
├── notebooks/                          # Jupyter notebooks for experiments
│   ├── rf_classification_experiments.ipynb
│   └── svr_regression_experiments.ipynb
│
├── .datasets/                          # Dataset download scripts (no raw data tracked)
│   └── download_datasets.py
│
├── assets/                             # Figures used in reports (plots, SHAP)
│   └── figures/
│
├── config.json                         # Experiment & model configuration
├── pyproject.toml                      # Poetry dependency configuration
├── .python-version                     # PyEnv Python version pin
├── README.md                           # Project documentation
└── LICENSE
```

---

## Experimental Highlights

- All custom models follow the **scikit-learn estimator API**
- Hyperparameter tuning via **Grid Search + Stratified K-Fold CV**
- Performance reported using:
  - Accuracy, Precision, Recall, F1-score (classification)
  - MAE, RMSE, R² (regression)
- Statistical analysis of cross-validation fold scores:
  - mean, variance, confidence intervals,
  - Shapiro–Wilk normality testing
- Model interpretability using **SHAP**:
  - global explanations (beeswarm, bar plots),
  - local explanations (waterfall plots),
  - feature effect analysis (dependence plots)

## Setup

This project uses `Poetry` as its package manager. For Poetry setup, refer to the [official website](https://python-poetry.org/).

To ensure robust Python version management, we use `PyEnv`. The Python version set for this project is set in the `.python-version` file. For instructions on how to setup PyEnv, refer to the [official repository](https://github.com/pyenv/pyenv).

To get you started, refer to the underlying pipeline:

- install the Poetry `shell` plugin if you do not already have it installed:

```cmd
poetry self add poetry-plugin-shell
```

- if you prefer having virtual environments created in the project root, execute this command:

```cmd
poetry config virtualenvs.in-project true
```

- install the project dependencies:

```cmd
poetry install
```

- activate the project environment:

```cmd
poetry shell
```

- run the dataset downloader:

```cmd
python ./.datasets/download_datasets.py
```

- setup done!

## Code Formatting

This project ships `black` as its default code formatter. To format the code, run the following command in the project root:

```cmd
poetry run python3 -m black <path>
```
