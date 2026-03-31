#  Fraud Detection System with EDA & Machine Learning

##  Overview

Fraudulent financial transactions pose a significant risk in digital banking systems. This project develops a machine learning-based fraud detection system that identifies suspicious transactions using both **data analysis** and **predictive modeling**.

The project follows an end-to-end pipeline:

> Data Exploration → Data Cleaning → Feature Analysis → Model Building → Deployment

---

##  Objectives

* Detect fraudulent transactions effectively
* Analyze transaction patterns using EDA
* Handle class imbalance in fraud datasets
* Build a reliable and deployable fraud detection model

---

##  Dataset

The dataset contains bank transaction records with both numerical and categorical features, including:

* Transaction details
* Customer-related attributes
* Transaction type
* Target variable:

  * `0` → Non-Fraud
  * `1` → Fraud

 The dataset is **highly imbalanced**, with fraudulent transactions forming a very small percentage of total transactions.

---

##  Exploratory Data Analysis (EDA)

A comprehensive EDA was performed to understand the dataset and uncover fraud patterns.

###  Class Imbalance

* Fraud cases are extremely rare compared to legitimate transactions
* This makes **accuracy an unreliable metric**
* Focus was placed on **Precision, Recall, and F1-score**

---

###  Numerical Feature Analysis

* Distribution of numerical features was analyzed
* Outliers were identified in transaction amounts
* Differences between fraud and non-fraud transactions were visualized
* Key numerical variables showed distinct fraud behavior patterns

---

###  Categorical Feature Analysis

* Fraud rates were analyzed across categorical variables
* High-risk categories were identified

####  Key Insight:

* Certain transaction types (e.g., transfers) showed **significantly higher fraud rates**
* Some categories contributed disproportionately to fraud occurrences

---

###  Feature Relationships

* Correlation analysis was performed
* Bivariate and multivariate analysis helped identify:

  * Strong predictors of fraud
  * Feature interactions

---

##  Data Preprocessing

* Missing values checked and handled
* Duplicate records removed
* Feature selection based on EDA insights
* Encoding applied to categorical variables
* Data prepared for machine learning models

---

##  Model Development

Machine learning models were trained to classify transactions as fraud or non-fraud.

Steps involved:

1. Splitting data into training and testing sets
2. Handling class imbalance (e.g., SMOTE or similar techniques)
3. Training models such as:

   * Logistic Regression
   * Random Forest *(adjust if needed)*
4. Evaluating model performance

---

##  Model Evaluation

Due to class imbalance, the following metrics were prioritized:

* **Precision** → Accuracy of fraud predictions
* **Recall** → Ability to detect actual fraud cases
* **F1-score** → Balance between precision and recall

> These metrics ensure the model is effective in identifying fraud without excessive false alarms.

---

##  Deployment

The trained model was deployed as an interactive application for real-time predictions.

* Users can input transaction details
* The system predicts whether the transaction is fraudulent

 **Live App:** **

---

##  Tech Stack

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Streamlit

---

##  Project Structure

```
fraud-detection-system/
│
├── data/                # Dataset (or sample)
├── notebooks/           # EDA and analysis
├── images/              # Saved visualizations
├── app.py               # Deployment script
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

---

##  Key Insights

* Fraud detection is a **highly imbalanced classification problem**
* Accuracy alone is misleading in fraud detection tasks
* EDA is critical in uncovering hidden fraud patterns
* Certain transaction types are significantly more prone to fraud
* Combining EDA with machine learning improves model reliability

---

##  What Makes This Project Stand Out

* Detailed step-by-step EDA with explanations
* Strong focus on real-world challenges (class imbalance)
* Clear interpretation of results
* End-to-end pipeline from analysis to deployment

---

##  Contact

Let’s connect:

* LinkedIn: 
* GitHub: 

---

 *This project demonstrates my ability to analyze complex datasets, extract meaningful insights, and build deployable machine learning solutions.*

