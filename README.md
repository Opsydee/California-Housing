# Project Section

# Project Title: House Price Prediction Using Random Forest Regression

# Project Overview:
This project focuses on predicting house prices in California based on various socio-economic and geographical features using the California Housing dataset. The model is developed using the Random Forest Regression algorithm, which provides high accuracy for complex, non-linear relationships in data. The goal is to predict the median house values based on features like median income, house age, average rooms, and geographical location.

# Table of Contents:
- Project Overview
- Installation and Setup
- Dataset
   - Source Data
   - Data Acquisition
   - Data Preprocessing
- Code Structure
- Usage
- Results and Evaluation
- Future Work
- Acknowledgments

# Installation and Setup:
- To set up this project, follow these steps:
      - Install Python and the necessary libraries:
         - import numpy as np
         - import pandas as pd
         - from sklearn.datasets import fetch_california_housing
         - from sklearn.model_selection import train_test_split
         - from sklearn.ensemble import RandomForestRegressor
         - from sklearn.metrics import mean_absolute_error

# Dataset:
 - Source Data: The project uses the California Housing dataset from the Scikit-Learn library.
 - Data Acquisition: The dataset is loaded using: h = fetch_california_housing()

# Data Preprocessing:
 - The dataset contains eight features and a target variable:
 Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
Target: MedHouseVal

# Code Structure:
   - data loading: Load and inspect the dataset
   - data preprocessing: Split the data into training and testing sets
   - model development: Train the Random Forest model
   - model evaluation: Evaluate using metrics like Mean Absolute Error (MAE) and R² Score

# Usage:
   - To train and evaluate the model, simply run the notebook cells in sequence:
      - RFR = RandomForestRegressor(random_state=2)
      - RFR.fit(d_train, t_train)
      - tp = RFR.predict(d_test)

# Results and Evaluation
     Mean Absolute Error (MAE): 0.336
     Mean Squared Error (MSE): 0.265
     Root Mean Squared Error (RMSE): 0.515
     R² Score: 0.804
The model performs well with an R² score of 0.804, indicating strong predictive capability.

# Future Work
   - Hyperparameter tuning for improved accuracy
   - Incorporation of additional features for better predictions
   - Exploration of other regression models for comparison
   - Deployment as a web-based application using frameworks like Streamlit

# Acknowledgments
   - Special thanks to:
     - Scikit-Learn for the California Housing dataset
     - Open-source community for continuous support
