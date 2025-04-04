# Air Quality Prediction Project

## Overview

This project aims to predict air quality levels using machine learning algorithms. It involves data exploration, preprocessing, model training, and evaluation. The dataset used contains various environmental factors that influence air quality.

## Code Structure and Logic

1. **Data Loading and Exploration:**
   - The code begins by importing necessary libraries like Pandas, Matplotlib, and Seaborn.
   - It loads the dataset from a CSV file using `pd.read_csv()`.
   - Exploratory data analysis is performed using `data.hist()` and `data.isnull().sum()` to understand the data distribution and missing values.
  
     ![image](https://github.com/user-attachments/assets/d3545a84-8590-4c5c-ae65-5f4a5ca1752a)


2. **Preprocessing:**
   - One-hot encoding is applied to the 'Air Quality' column using `pd.get_dummies()` to convert categorical values into numerical representations.
   - The dataset is split into training and testing sets using `train_test_split()` from scikit-learn.

3. **Model Training and Evaluation:**
   - Several machine learning models are trained, including:
     - Gradient Boosting Classifier
     - AdaBoost Classifier
     - XGBoost Classifier
     - LightGBM Classifier
     - Stacking Classifier (combining the above models)
   - Each model is trained using its respective library and the training data.
   - Model performance is evaluated using metrics like accuracy and classification report.

4. **Feature Importance:**
   - The importance of each feature in the Gradient Boosting model is visualized using `sns.barplot()`.

5. **Model Comparison:**
   - The accuracies of all trained models are compared using a bar plot.
  ![image](https://github.com/user-attachments/assets/5d5718c1-60f1-4acb-9995-2303f2795bf5)


## Technology and Algorithms

- **Python:** The primary programming language used.
- **Pandas:** For data manipulation and analysis.
- **Matplotlib and Seaborn:** For data visualization.
- **Scikit-learn:** For model selection, training, and evaluation.
- **Gradient Boosting, AdaBoost, XGBoost, LightGBM:** Ensemble learning algorithms used for prediction.
- **Stacking:** A technique to combine multiple models for improved performance.

## Usage

1. Ensure you have the required libraries installed: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`.
2. Load the dataset into a Pandas DataFrame.
3. Run the code cells sequentially to perform data preprocessing, model training, and evaluation.
4. Analyze the results and visualizations to understand the model's performance.

## Conclusion

This project demonstrates the application of machine learning for air quality prediction. The results highlight the effectiveness of ensemble methods in achieving high accuracy. Further improvements can be explored by experimenting with hyperparameter tuning and feature engineering.
