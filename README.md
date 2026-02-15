# MachineLearningAssignment2

# Credit Score Multi-Class Classification

A machine learning project for predicting credit scores (Good, Standard, Poor) using multiple classification algorithms with a Streamlit web application for easy model deployment and predictions.

## 📁 Project Structure

```
MachineLearningAssignment2/
│
├── dataset/
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
│
├── model/
│   └── model.ipynb         # Jupyter notebook for model training
│
├── saved_models/           # Directory for saved models (created after training)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   ├── feature_names.pkl
│   └── model_results.csv
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Training Models

### Step 1: Train the Models in Jupyter Notebook

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook model/model.ipynb
   ```

2. Run all cells in the notebook sequentially:
   - Data loading and exploration
   - Data preprocessing and feature engineering
   - Model training (6 models)
   - Model evaluation and comparison
   - **Model saving** (final cells)

3. The trained models will be saved in the `saved_models/` directory

### Models Trained:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest
6. XGBoost

### Evaluation Metrics:
- Accuracy
- ROC AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 🌐 Running the Streamlit App

### Step 2: Launch the Web Application

After training and saving the models, run:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📱 Using the Streamlit App

### Features:

1. **Model Selection Dropdown** (Sidebar)
   - Choose from 6 trained classification models
   - Each model's performance can be compared

2. **Dataset Upload** (Sidebar)
   - Upload test data in CSV format
   - Supports files with or without labels
   - Automatically preprocesses uploaded data

3. **Predictions Display**
   - Shows predicted credit scores
   - Displays probability scores for each class
   - Download predictions as CSV

4. **Evaluation Metrics** (if labels provided)
   - Accuracy, Precision, Recall
   - F1 Score, ROC AUC, MCC Score
   - Displayed in easy-to-read metric cards

5. **Confusion Matrix**
   - Visual heatmap representation
   - Shows prediction accuracy per class

6. **Classification Report**
   - Detailed per-class metrics
   - Color-coded performance indicators

7. **Prediction Distribution**
   - Comparison of actual vs predicted distributions
   - Bar chart visualizations

## 📝 Data Preprocessing

The app automatically applies the same preprocessing steps as training:

- **Handling missing values**: Median imputation for numeric, mode/Unknown for categorical
- **Outlier treatment**: Capping extreme values (Age, Credit Inquiries, etc.)
- **Feature engineering**: One-hot encoding for Type_of_Loan, parsing Credit_History_Age
- **Data cleaning**: Removing special characters, underscores
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler normalization

## 🎯 Expected Test Data Format

Upload a CSV file with the following columns (Credit_Score is optional for prediction-only):

- Age, Occupation, Annual_Income, Monthly_Inhand_Salary
- Num_Bank_Accounts, Num_Credit_Card, Interest_Rate
- Num_of_Loan, Type_of_Loan, Delay_from_due_date
- Num_of_Delayed_Payment, Changed_Credit_Limit
- Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt
- Credit_Utilization_Ratio, Credit_History_Age
- Payment_of_Min_Amount, Total_EMI_per_month
- Amount_invested_monthly, Payment_Behaviour
- Monthly_Balance
- **Credit_Score** (optional - for evaluation)

## 🔧 Troubleshooting

**Models not loading?**
- Ensure you've run the model training notebook completely
- Check that `saved_models/` directory exists and contains `.pkl` files

**Upload error?**
- Verify CSV format matches expected columns
- Check for encoding issues (use UTF-8)

**Preprocessing errors?**
- Ensure column names match exactly
- Check for unexpected data formats

## 📦 Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning algorithms and metrics
- xgboost: Gradient boosting classifier
- streamlit: Web application framework
- matplotlib: Visualization
- seaborn: Statistical visualizations

## 👨‍💻 Development

To modify or extend the project:

1. **Add new models**: Train in notebook, save with pickle, add to `models_dict` in app.py
2. **Modify preprocessing**: Update both notebook and `preprocess_data()` function
3. **Customize UI**: Edit `app.py` Streamlit components

## 📈 Model Performance

After training, check `saved_models/model_results.csv` for comparative performance metrics of all models.

## 🎓 Assignment Requirements Checklist

- ✅ Dataset upload option (CSV)
- ✅ Model selection dropdown
- ✅ Display of evaluation metrics
- ✅ Confusion matrix visualization
- ✅ Classification report
- ✅ Multiple models comparison
- ✅ Professional UI/UX

## 📄 License

This project is for educational purposes.