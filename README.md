# Credit Score Multi-Class Classification

## a. Problem Statement

The objective of this project is to predict credit scores (Good, Standard, Poor) based on various financial and personal attributes of customers. Credit score prediction is crucial for financial institutions to assess the creditworthiness of individuals and make informed lending decisions. This multi-class classification problem uses machine learning algorithms to analyze historical customer data and predict credit score categories.

![UI](<Screenshot 2026-02-15 191116.png>)
![UI](<Screenshot 2026-02-15 191153.png>)

## b. Dataset Description

The dataset contains customer financial and credit history information with the following characteristics:

- **Training Data**: 100,000 records with 28 features
- **Test Data**: 50,000 records
- **Target Variable**: Credit_Score (3 classes: Good, Standard, Poor)

**Key Features Include**:
- **Personal Information**: Age, Occupation
- **Financial Metrics**: Annual_Income, Monthly_Inhand_Salary, Monthly_Balance
- **Credit Behavior**: Num_Bank_Accounts, Num_Credit_Card, Num_of_Loan, Type_of_Loan
- **Payment Patterns**: Num_of_Delayed_Payment, Delay_from_due_date, Payment_Behaviour, Payment_of_Min_Amount
- **Credit History**: Credit_History_Age, Credit_Mix, Credit_Utilization_Ratio
- **Loan Details**: Outstanding_Debt, Total_EMI_per_month, Amount_invested_monthly, Interest_Rate

**Data Preprocessing Applied**:
- Missing value imputation (median for numeric, mode/Unknown for categorical)
- Outlier treatment (Age capping, Credit Inquiries capping)
- Feature engineering (One-hot encoding for Type_of_Loan, Credit_History_Age parsing)
- Data cleaning (removing underscores, special characters)
- Label encoding for categorical variables
- StandardScaler normalization for numerical features

## c. Models Used

Six classification algorithms were trained and evaluated using the following metrics: Accuracy, ROC AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Model Comparison Table

| ML Model Name              | Accuracy | ROC AUC | Precision | Recall | F1 Score | MCC    |
|----------------------------|----------|---------|-----------|--------|----------|--------|
| Logistic Regression        | 0.6276   | 0.7567  | 0.6247    | 0.6276 | 0.6182   | 0.3491 |
| Decision Tree              | 0.6888   | 0.7338  | 0.6895    | 0.6888 | 0.6891   | 0.4811 |
| K-Nearest Neighbors        | 0.7048   | 0.8283  | 0.7045    | 0.7048 | 0.7046   | 0.5058 |
| Naive Bayes (Gaussian)     | 0.5068   | 0.6931  | 0.5960    | 0.5068 | 0.4945   | 0.3259 |
| Random Forest (Ensemble)   | 0.7956   | 0.9010  | 0.7956    | 0.7956 | 0.7955   | 0.6590 |
| XGBoost (Ensemble)         | 0.7546   | 0.8731  | 0.7554    | 0.7546 | 0.7549   | 0.5904 |

**Best Model by Accuracy**: Random Forest (79.56%)

### Model Performance Observations

| ML Model Name              | Observation about Model Performance |
|----------------------------|-------------------------------------|
| Logistic Regression        | Shows moderate performance (62.76% accuracy) as a baseline linear model. Limited ability to capture non-linear relationships in credit data. Better suited for linearly separable problems. ROC AUC of 0.7567 indicates reasonable class separation capability. |
| Decision Tree              | Achieves 68.88% accuracy with good interpretability. Prone to overfitting on training data. Lower ROC AUC (0.7338) compared to ensemble methods suggests limited generalization. Useful for understanding feature importance and decision rules. |
| K-Nearest Neighbors        | Performs reasonably well (70.48% accuracy) with good ROC AUC (0.8283). Distance-based approach works well with normalized features. Computationally expensive for large datasets. Sensitive to feature scaling, which was addressed through StandardScaler. |
| Naive Bayes (Gaussian)     | Poorest performance (50.68% accuracy) among all models. Assumes feature independence, which may not hold for financial data where features are often correlated. Fast training time but limited predictive power. Not recommended for this dataset. |
| Random Forest (Ensemble)   | Best overall performance (79.56% accuracy, 0.9010 ROC AUC). Excellent at handling non-linear relationships and feature interactions. Robust to outliers and noise. Ensemble of 20 decision trees with max_depth=15 balances performance and model size. High MCC (0.6590) indicates strong predictive capability across all classes. |
| XGBoost (Ensemble)         | Second-best performance (75.46% accuracy, 0.8731 ROC AUC). Gradient boosting provides strong predictive power through sequential learning. Handles imbalanced classes effectively. Slightly lower than Random Forest but offers faster prediction time. Good balance between accuracy and computational efficiency. |

## Project Structure

```
MachineLearningAssignment2/
│
├── dataset/
│   ├── train.csv              # Training dataset
│   └── test.csv               # Test dataset
│
├── model/
│   └── model.ipynb            # Jupyter notebook for model training
│
├── saved_models/              # Trained models and preprocessing objects
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── model_results.csv
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
jupyter notebook model/model.ipynb
```
Run all cells in the notebook to train models and save them to `saved_models/` directory.

### 3. Launch Streamlit App
```bash
streamlit run app.py
```
Access the web application at `http://localhost:8501`

## Streamlit App Features

- **Model Selection**: Choose from 6 trained classification models
- **CSV Upload**: Upload test data for predictions
- **Predictions**: View predicted credit scores with confidence probabilities
- **Evaluation Metrics**: Display accuracy, precision, recall, F1, ROC AUC, MCC (if labels provided)
- **Confusion Matrix**: Visual heatmap of prediction accuracy
- **Classification Report**: Detailed per-class performance metrics
- **Download Results**: Export predictions as CSV file

## Technologies Used

- **Python 3.11+**
- **scikit-learn 1.7.2**: Machine learning algorithms
- **XGBoost 3.2.0**: Gradient boosting classifier
- **Streamlit 1.30+**: Web application framework
- **pandas, numpy**: Data manipulation
- **matplotlib, seaborn**: Visualization