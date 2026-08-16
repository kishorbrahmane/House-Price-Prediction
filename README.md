# House Price Prediction

A machine learning project to predict median house values using the California Housing Prices dataset.

## Project Overview

This project follows an end-to-end regression workflow:

**EDA → Preprocessing → Baseline Model → Cross-Validation → Model Selection → Hyperparameter Tuning → Final Evaluation → Prediction**

The goal is to build and evaluate regression models while maintaining a clean preprocessing pipeline and avoiding data leakage.

## Dataset

**California Housing Prices**

* Source: Kaggle
* Dataset: `housing.csv`
* Target variable: `median_house_value`

### Features

* `longitude`
* `latitude`
* `housing_median_age`
* `total_rooms`
* `total_bedrooms`
* `population`
* `households`
* `median_income`
* `ocean_proximity`

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

### 1. Exploratory Data Analysis

The dataset was examined to understand:

* Feature distributions
* Missing values
* Outliers and skewness
* Relationships between features and house prices
* Correlations between numerical variables

Key observations included:

* `total_bedrooms` contains missing values.
* `median_house_value` is right-skewed and capped.
* `median_income` is a strong predictor of house value.
* Several room and population features show high multicollinearity.

### 2. Data Preprocessing

A Scikit-learn preprocessing pipeline was used to ensure consistent transformations.

* Median imputation for missing numerical values
* One-hot encoding for `ocean_proximity`
* Feature scaling for linear models
* Pipeline-based preprocessing to reduce data leakage

### 3. Baseline Model

**Linear Regression** was used as the baseline model.

The baseline was evaluated using:

* RMSE
* MAE
* R²

### 4. Model Comparison

The following regression models were compared using cross-validation:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest
* HistGradientBoostingRegressor

The primary model selection metric was **RMSE**, with MAE and R² used as additional evaluation metrics.

### 5. Best Model

**HistGradientBoostingRegressor** achieved the best cross-validation performance based on RMSE and was selected for further optimization.

### 6. Hyperparameter Tuning

`GridSearchCV` was used to tune the HistGradientBoostingRegressor.

The final model used the selected parameters for:

* Learning rate
* Maximum depth
* Maximum leaf nodes
* Minimum samples per leaf
* L2 regularization

### 7. Final Evaluation

The tuned model was retrained and evaluated on the held-out test set using:

* **RMSE**
* **MAE**
* **R²**

The test set was kept separate until final evaluation.

## Prediction

A prediction pipeline was created to estimate the median house value for new observations using the trained preprocessing and regression model.

## Future Improvements

Potential improvements include:

* Creating ratio features such as rooms per household
* Creating bedrooms-per-room and population-per-household features
* Applying transformations to the target variable
* Further tuning HistGradientBoosting parameters
* Experimenting with XGBoost or LightGBM
* Performing error analysis across different locations and price ranges
* Exploring spatial or grouped validation

## Project Structure

```text
house-price-prediction/
│
├── housing.csv
├── house_price_prediction.ipynb
└── README.md
```

## How to Run

### Clone the repository

```bash
git clone <repository-url>
cd house-price-prediction
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the notebook

Open:

```text
house_price_prediction.ipynb
```

and execute the cells sequentially.

## Conclusion

This project demonstrates a complete machine learning regression workflow, from exploratory analysis and preprocessing to model comparison, cross-validation, hyperparameter tuning, and final evaluation.
