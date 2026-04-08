# Food Nutrition Density Prediction

## Overview
This notebook demonstrates a machine learning pipeline to predict the 'Nutrition Density' of various food items based on their nutritional components. It involves data loading, preprocessing, model training with several regression algorithms, and evaluation of their performance.

## Problem Statement
The goal is to build predictive models that can accurately estimate the 'Nutrition Density' of food items using a comprehensive set of nutrient data. This can help in understanding which nutritional components contribute most to a food's overall nutritional density.

## Dataset
The dataset consists of multiple CSV files containing detailed nutritional information for a wide range of food items. These files were combined into a single DataFrame. Key features include:
- `food`: Name of the food item (categorical).
- `Caloric Value`
- `Fat`, `Saturated Fats`, `Monounsaturated Fats`, `Polyunsaturated Fats`
- `Carbohydrates`, `Sugars`, `Protein`, `Dietary Fiber`
- `Cholesterol`, `Sodium`, `Water`
- Various `Vitamins` (A, B1, B11, B12, B2, B3, B5, B6, C, D, E, K)
- `Minerals` (Calcium, Copper, Iron, Magnesium, Manganese, Phosphorus, Potassium, Selenium, Zinc)
- `Nutrition Density`: The target variable for prediction.

## Methodology

### 1. Data Loading and Combination
Multiple CSV files (e.g., `FOOD-DATA-GROUP1.csv`, `FOOD-DATA-GROUP2.csv`, `FOOD-DATA-GROUP3.csv`) were uploaded and concatenated into a single Pandas DataFrame (`combined_df`).

### 2. Data Preprocessing
- **Column Dropping**: Redundant index columns (`Unnamed: 0.1`, `Unnamed: 0`) were removed.
- **Feature and Target Split**: The dataset was divided into features (`X`) and the target variable (`y`), which is 'Nutrition Density'.
- **Train-Test Split**: The data was split into training (80%) and testing (20%) sets to evaluate model generalization (`X_train`, `X_test`, `y_train`, `y_test`).

### 3. Feature Engineering and Preprocessing Pipeline
A `ColumnTransformer` was used to apply different preprocessing steps to numerical and categorical features:
- **Numerical Features**: Imputed missing values with the mean (`SimpleImputer`) and scaled using `StandardScaler`.
- **Categorical Features**: Imputed missing values with the most frequent value (`SimpleImputer`) and encoded using `OneHotEncoder`.

### 4. Model Training and Evaluation
Three different regression models were trained and evaluated:
- **Linear Regression**
- **Random Forest Regressor**
- **Support Vector Regressor (SVR)**

Each model was integrated into a `Pipeline` with the preprocessor. Performance was assessed using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R2)**

### 5. Visualizations
- **Scatter Plot**: Visualized the relationship between 'Protein' content and 'Nutrition Density'.
- **Heatmap**: Displayed the Pearson correlation coefficients between numerical nutrients and 'Nutrition Density' to identify strong relationships.
- **Model Performance Comparison Graphs**: Bar plots comparing R2, MSE, and MAE across the trained models.

## Results and Model Comparison

The models yielded the following performance metrics on the test set:

- **Linear Regression**:
  - R2: 1.00
  - MSE: 0.35
  - MAE: 0.28

- **Random Forest Regressor**:
  - R2: 0.91
  - MSE: 3706.24
  - MAE: 17.06

- **Support Vector Regressor (SVR)**:
  - R2: 0.18
  - MSE: 33271.82
  - MAE: 74.64

**Linear Regression** performed exceptionally well with an R2 of 1.00 and very low MSE/MAE, suggesting a strong linear relationship between the features and the target, or potentially some data leakage which might need further investigation. The **Random Forest Regressor** also showed good performance (R2 of 0.91), while the **Support Vector Regressor** had the lowest performance among the three models tested.

Further analysis could involve hyperparameter tuning, feature selection, or exploring other advanced regression techniques to potentially improve the models, especially for the Random Forest and SVR.
