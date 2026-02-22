Medical Cost Prediction using Machine Learning
Project Overview

This project predicts medical insurance charges based on personal and demographic features such as:

Age

Sex

BMI

Number of Children

Smoking Status

Region

The goal is to build a regression model that estimates healthcare costs accurately and analyzes how factors like smoking impact medical expenses.

Problem Statement

Insurance companies need to estimate medical costs for individuals to determine premiums.
This project uses machine learning regression techniques to predict insurance charges based on historical data.

Dataset Information

Dataset Name: Medical Insurance Cost Dataset

Common Source: Kaggle

Records: Approximately 1300+ rows

Target Variable: charges

Features:
Feature	Description
age	Age of the person
sex	Gender
bmi	Body Mass Index
children	Number of dependents
smoker	Yes/No
region	Residential region
charges	Medical cost (Target Variable)
Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Machine Learning Models Used

Linear Regression

Ridge / Lasso Regression

Random Forest Regressor

Project Workflow

Data Collection

Data Cleaning

Exploratory Data Analysis (EDA)

Feature Encoding

Train-Test Split

Model Training

Model Evaluation

Performance Comparison

Model Evaluation Metrics

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Sample Results
Model	R² Score
Linear Regression	~0.75
Random Forest	~0.85

(Scores may vary based on tuning)

Key Insights

Smokers have significantly higher medical costs.

BMI and Age strongly influence insurance charges.

Random Forest performed better than Linear Regression.

Project Structure
Medical-Cost-Prediction/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   └── EDA_and_Model.ipynb
│
├── src/
│   └── model.py
│
├── requirements.txt
└── README.md
Future Improvements

Hyperparameter tuning

Deploy model using Flask or Streamlit

Add cross-validation

Use advanced models (XGBoost, Gradient Boosting)

Author

BIPLAV PARTHASARATHI ROUT
LinkedIn: https://www.linkedin.com/in/biplav-parthasarathi-rout-673a10352/
GitHub: pastrystrawberry



