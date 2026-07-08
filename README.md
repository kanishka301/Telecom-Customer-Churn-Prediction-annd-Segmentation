# Telecom Customer Churn Prediction

## Overview

This project predicts the likelihood of customer churn in a telecom company using machine learning and provides an interactive Streamlit dashboard for customer-level predictions, exploratory data analysis, and customer segmentation.

The project includes:

* A complete data preprocessing and feature engineering pipeline.
* A machine learning model for customer churn prediction.
* Customer segmentation based on churn probability and customer characteristics.
* An interactive Streamlit dashboard for business users.

---

## Business Impact

Customer churn is one of the biggest challenges for subscription-based businesses. This project enables telecom companies to proactively identify customers who are likely to leave and take appropriate retention actions before churn occurs.

The project helps businesses:

* Predict churn risk for individual customers.
* Understand customer behavior through exploratory analysis.
* Segment customers into meaningful groups for targeted marketing.
* Improve customer retention while reducing acquisition costs.

---

# Project Structure

```
Telecom Churn Prediction/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── cleaned.csv
│   ├── churn_prob.csv
│   └── Final_Segments.csv
│
├── model/
│   └── model_bundle.pkl
│
└── notebook/
    ├── Data Preprocessing.ipynb
    ├── Model_Training.ipynb
    └── segmentation.ipynb
```

---

# Files & Key Components

### app.py

The Streamlit dashboard includes three major sections:

### 1. Predict Churn

* Predicts churn for a single customer.
* Loads the trained model and preprocessing objects from `model/model_bundle.pkl`.
* Uses the same encoders and scaler used during training.
* Displays:

  * Churn prediction
  * Churn probability
  * Risk progress bar

---

### 2. Explore Data

Provides exploratory analysis including:

* Total customers
* Churn rate
* Customer distribution
* Tenure distribution
* Churn by contract type
* Interactive visualizations
* Raw dataset viewer

---

### 3. Customer Segments

Displays customer segmentation results including:

* Number of customers in each segment
* Segment distribution
* Average customer profile for each segment
* Mean tenure
* Monthly charges
* Total charges
* Average churn probability

---

## Notebooks

### Data Preprocessing.ipynb

Responsible for:

* Cleaning missing values
* Encoding categorical variables
* Feature engineering
* Data preprocessing

---

### Model_Training.ipynb

Responsible for:

* Model training
* Feature scaling
* Label encoding
* Model evaluation
* Saving the trained model bundle

Output:

```
model/model_bundle.pkl
```

---

### segmentation.ipynb

Responsible for:

* Customer segmentation
* Calculating churn probabilities
* Creating customer segments
* Exporting:

```
data/Final_Segments.csv
```

---

# Data

### Original Dataset

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Contains telecom customer information including:

* Customer demographics
* Services subscribed
* Billing information
* Contract details
* Churn status

---

### Processed Files

* `cleaned.csv`
* `churn_prob.csv`
* `Final_Segments.csv`

These files are generated during preprocessing, prediction, and segmentation.

---

# Model

The application loads a serialized model bundle containing:

* Trained machine learning model
* Feature scaler
* Label encoders
* Feature names

This ensures that prediction-time preprocessing exactly matches the training pipeline.

---

# How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the notebooks

Execute the notebooks in the following order:

1. `Data Preprocessing.ipynb`
2. `Model_Training.ipynb`
3. `segmentation.ipynb`

This generates:

* Cleaned datasets
* Customer segments
* Trained model

---

# Dashboard Features

### Predict Churn

* Interactive customer form
* Real-time churn prediction
* Churn probability
* Risk indicator

---

### Explore Data

* Total customers
* Churned customers
* Overall churn rate
* Churn distribution
* Tenure analysis
* Contract type analysis
* Raw dataset explorer

---

### Customer Segments

* Customer count by segment
* Segment distribution
* Average customer characteristics
* Average churn probability per segment

---

# Key Findings

* Customers with **month-to-month contracts** are more likely to churn.
* Customers with **short tenure** generally exhibit higher churn rates.
* Higher **monthly charges** are associated with increased churn risk.
* Customer segmentation helps identify groups with similar characteristics and churn behavior.
* The dashboard enables quick identification of high-risk customers for proactive retention efforts.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Streamlit
* Pickle

---

# Future Improvements

* Compare multiple machine learning models (Random Forest, XGBoost, LightGBM, CatBoost).
* Hyperparameter tuning.
* SHAP-based feature importance.
* Retention recommendation engine.
* Cloud deployment.
* Real-time prediction API.
* Automated retraining pipeline.

---

