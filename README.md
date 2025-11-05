# Customer Churn Prediction

## Overview

Customer churn, the loss of customers over time, poses a significant challenge for businesses, especially in competitive sectors like telecommunications. This project develops a machine learning model using real-world telecom data to predict which customers are likely to churn.

The model identifies critical factors influencing churn—such as contract details, tenure, payment methods, and usage charges—and predicts customer behavior with high accuracy. It is deployed as an interactive Flask web application enabling users to input customer attributes and obtain churn likelihood predictions instantly.

## Features

- Data cleaning and preprocessing of telecom customer dataset.
- Identification of top features influencing churn via Random Forest model.
- Training and saving a predictive model on selected features.
- Flask-based web interface with user-friendly dropdowns and inputs.
- Real-time churn prediction with clear, actionable output.

## Technologies Used

- Python, scikit-learn, pandas, joblib
- Flask web framework
- HTML/CSS for frontend forms

## Getting Started

1. Clone this repository:
   git clone <your-repo-url>
   cd <your-project-folder>

2. Install dependencies:
   pip install -r requirements.txt

3. Run the training script to generate the model and feature list:
   python src/train_model.py
 
4. Launch the Flask app:
   python src/app.py

