---
title: Customer Churn Predictor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: python
app_file: app.py
---
# Customer Churn Prediction Web App

This is a full-stack data science project that predicts customer churn based on their account information. The project is built with a Scikit-learn model and deployed as a web application using Flask.

## Features

-   Predicts customer churn probability using a trained Random Forest model.
-   Simple, clean web interface for user input.
-   Real-time prediction results displayed on the front-end.

## Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, Pandas, NumPy
-   **Frontend:** HTML, CSS, JavaScript
-   **Dataset:** [Telco Customer Churn from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ahnaf-Shahadat-Taseen/customer-churn-predictor.git
    cd customer-churn-predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    flask run
    ```
5.  Open your web browser and navigate to `http://127.0.0.1:5000`.
