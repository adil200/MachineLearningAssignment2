import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, confusion_matrix, 
                             classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #333333 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    h1 {
        color: #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #b3d9ed;
    }
    .info-box h3 {
        color: #0066cc;
        margin-top: 0;
    }
    .info-box p {
        color: #333333;
        margin-bottom: 0;
    }
    .step-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        height: 100%;
    }
    .step-card h3 {
        color: #0066cc;
        font-size: 1.2em;
    }
    .step-card p {
        color: #333333;
        font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Credit Score Multi-Class Classification")
st.markdown("""
<div class='info-box'>
    <h3>Model Prediction System</h3>
    <p>Upload your test dataset, select a machine learning model, and get instant predictions with comprehensive evaluation metrics.</p>
</div>
""", unsafe_allow_html=True)

# Function to load models and preprocessing objects
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes (Gaussian)': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        filepath = f'saved_models/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # Load preprocessing objects
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('saved_models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('saved_models/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    
    with open('saved_models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return models, scaler, label_encoders, target_encoder, feature_names

# Function to preprocess uploaded data
def preprocess_data(df, label_encoders, feature_names):
    """
    Apply the same preprocessing steps as training data
    """
    df = df.copy()
    
    # Drop ID columns if present
    id_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
    df = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
    
    # Age preprocessing
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df = df[df['Age'] <= 110]
        df.loc[df['Age'] < 18, 'Age'] = 18
    
    # Occupation preprocessing
    if 'Occupation' in df.columns:
        most_common = df[df['Occupation'] != '_______']['Occupation'].mode()
        if len(most_common) > 0:
            df['Occupation'] = df['Occupation'].replace('_______', most_common[0])
    
    # Numeric columns - remove underscores and convert
    numeric_cols = ['Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                   'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 
                   'Monthly_Balance']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Handle Type_of_Loan - create binary features
    if 'Type_of_Loan' in df.columns:
        df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Not Specified')
        
        loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan', 'Home Equity Loan',
                     'Mortgage Loan', 'Student Loan', 'Debt Consolidation Loan', 'Payday Loan',
                     'Not Specified']
        
        for loan_type in loan_types:
            column_name = f'Has_{loan_type.replace(" ", "_").replace("-", "_")}'
            df[column_name] = df['Type_of_Loan'].str.contains(loan_type, case=False, na=False).astype(int)
        
        df['Num_Loan_Types'] = df['Type_of_Loan'].apply(
            lambda x: len(x.split(',')) if isinstance(x, str) and x != 'Not Specified' else 0
        )
        df = df.drop('Type_of_Loan', axis=1)
    
    # Other numeric columns
    if 'Monthly_Inhand_Salary' in df.columns:
        df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(df['Monthly_Inhand_Salary'].median())
    
    if 'Num_Credit_Inquiries' in df.columns:
        df['Num_Credit_Inquiries'] = pd.to_numeric(df['Num_Credit_Inquiries'], errors='coerce')
        df.loc[df['Num_Credit_Inquiries'] > 20, 'Num_Credit_Inquiries'] = 20
        df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].fillna(df['Num_Credit_Inquiries'].median())
    
    if 'Num_Bank_Accounts' in df.columns:
        df['Num_Bank_Accounts'] = pd.to_numeric(df['Num_Bank_Accounts'], errors='coerce')
        df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].fillna(df['Num_Bank_Accounts'].median())
    
    if 'Num_Credit_Card' in df.columns:
        df['Num_Credit_Card'] = pd.to_numeric(df['Num_Credit_Card'], errors='coerce')
        df.loc[df['Num_Credit_Card'] > 15, 'Num_Credit_Card'] = 15
        df['Num_Credit_Card'] = df['Num_Credit_Card'].fillna(df['Num_Credit_Card'].median())
    
    if 'Interest_Rate' in df.columns:
        df['Interest_Rate'] = pd.to_numeric(df['Interest_Rate'], errors='coerce')
        df.loc[df['Interest_Rate'] < 0, 'Interest_Rate'] = df['Interest_Rate'].median()
        df['Interest_Rate'] = df['Interest_Rate'].fillna(df['Interest_Rate'].median())
    
    if 'Delay_from_due_date' in df.columns:
        df['Delay_from_due_date'] = pd.to_numeric(df['Delay_from_due_date'], errors='coerce')
        df['Delay_from_due_date'] = df['Delay_from_due_date'].fillna(df['Delay_from_due_date'].median())
    
    # Credit_Mix
    if 'Credit_Mix' in df.columns:
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', 'Unknown')
        df['Credit_Mix'] = df['Credit_Mix'].fillna('Unknown')
    
    if 'Credit_Utilization_Ratio' in df.columns:
        df['Credit_Utilization_Ratio'] = pd.to_numeric(df['Credit_Utilization_Ratio'], errors='coerce')
        df['Credit_Utilization_Ratio'] = df['Credit_Utilization_Ratio'].fillna(df['Credit_Utilization_Ratio'].median())
    
    # Credit_History_Age parsing
    if 'Credit_History_Age' in df.columns:
        years = df['Credit_History_Age'].astype(str).str.extract(r'(\d+)\s*Years?', expand=False).astype(float)
        months = df['Credit_History_Age'].astype(str).str.extract(r'(\d+)\s*Months?', expand=False).astype(float)
        df['Credit_History_Age'] = years.fillna(0) * 12 + months.fillna(0)
        df['Credit_History_Age'] = df['Credit_History_Age'].fillna(df['Credit_History_Age'].median())
    
    # Payment_of_Min_Amount
    if 'Payment_of_Min_Amount' in df.columns:
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', 'Unknown')
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].fillna('Unknown')
    
    if 'Total_EMI_per_month' in df.columns:
        df['Total_EMI_per_month'] = pd.to_numeric(df['Total_EMI_per_month'], errors='coerce')
        df['Total_EMI_per_month'] = df['Total_EMI_per_month'].fillna(df['Total_EMI_per_month'].median())
    
    # Payment_Behaviour
    if 'Payment_Behaviour' in df.columns:
        def clean_payment_behaviour(value):
            if pd.isna(value):
                return 'Unknown'
            if not any(word in str(value) for word in ['spent', 'value', 'payments', 'Low', 'High', 'Small', 'Large', 'Medium']):
                return 'Unknown'
            return value
        df['Payment_Behaviour'] = df['Payment_Behaviour'].apply(clean_payment_behaviour)
    
    # Label encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'Credit_Score' in categorical_columns:
        categorical_columns.remove('Credit_Score')
    
    for col in categorical_columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        else:
            # If encoder doesn't exist, create a simple numeric encoding
            df[col] = pd.factorize(df[col])[0]
    
    # Ensure all feature columns match training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features used in training
    df = df[feature_names]
    
    return df

# Sidebar
st.sidebar.header("Configuration")

# Load models
@st.cache_resource
def load_all_models():
    """Load all models and preprocessing objects"""
    try:
        models = {}
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbors': 'knn.pkl',
            'Naive Bayes (Gaussian)': 'naive_bayes.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = f'saved_models/{filename}'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
        
        # Load preprocessing objects
        with open('saved_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('saved_models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('saved_models/target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        
        with open('saved_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return models, scaler, label_encoders, target_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please make sure you have trained the models first by running all cells in model.ipynb")
        return None, None, None, None, None

# Load models
models, scaler, label_encoders, target_encoder, feature_names = load_all_models()

if models is None:
    st.stop()

st.sidebar.success(f"{len(models)} models loaded successfully")

# Model selection dropdown
st.sidebar.markdown("### Select Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys()),
    help="Select from 6 trained machine learning models"
)

# File upload
st.sidebar.markdown("### Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your test dataset (CSV format)"
)

# Only show predict button if file is uploaded
if uploaded_file is not None:
    st.sidebar.markdown("### Ready to Predict")
    predict_button = st.sidebar.button("Predict & Evaluate", type="primary", width="stretch")
else:
    predict_button = False
    st.sidebar.info("Please upload a CSV file to continue")

# Main content
if uploaded_file is not None:
    # Load and display data preview
    test_data = pd.read_csv(uploaded_file)
    
    with st.expander("Dataset Preview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", test_data.shape[0])
        with col2:
            st.metric("Total Columns", test_data.shape[1])
        with col3:
            has_labels = 'Credit_Score' in test_data.columns
            st.metric("Has Labels", "Yes" if has_labels else "No")
        
        st.dataframe(test_data.head(10), width="stretch")
    
    # Check if predict button is clicked
    if predict_button:
        with st.spinner("Processing data and making predictions..."):
            try:
                # Preprocess data
                X_test = preprocess_data(test_data.copy(), label_encoders, feature_names)
                X_test_scaled = scaler.transform(X_test)
                
                # Convert back to DataFrame to preserve feature names
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
                
                # Make predictions
                selected_model = models[selected_model_name]
                
                if selected_model_name == 'XGBoost':
                    y_pred_encoded = selected_model.predict(X_test_scaled)
                    y_pred = target_encoder.inverse_transform(y_pred_encoded)
                    y_pred_proba = selected_model.predict_proba(X_test_scaled)
                else:
                    y_pred = selected_model.predict(X_test_scaled)
                    y_pred_proba = selected_model.predict_proba(X_test_scaled)
                
                st.success("Predictions completed successfully")
                
                # Display predictions
                st.markdown("---")
                st.subheader(f"Predictions using {selected_model_name}")
                
                predictions_df = pd.DataFrame({
                    'Predicted_Credit_Score': y_pred,
                    'Confidence_Good': [f"{p:.2%}" for p in y_pred_proba[:, np.where(selected_model.classes_ == 'Good')[0][0]]] if 'Good' in selected_model.classes_ else [f"{p:.2%}" for p in y_pred_proba[:, 0]],
                    'Confidence_Standard': [f"{p:.2%}" for p in y_pred_proba[:, np.where(selected_model.classes_ == 'Standard')[0][0]]] if 'Standard' in selected_model.classes_ else [f"{p:.2%}" for p in y_pred_proba[:, 1]],
                    'Confidence_Poor': [f"{p:.2%}" for p in y_pred_proba[:, np.where(selected_model.classes_ == 'Poor')[0][0]]] if 'Poor' in selected_model.classes_ else [f"{p:.2%}" for p in y_pred_proba[:, 2]]
                })
                
                if has_labels:
                    y_true = test_data['Credit_Score'].iloc[:len(y_pred)]
                    predictions_df.insert(0, 'Actual_Credit_Score', y_true.values)
                
                st.dataframe(predictions_df.head(20), width="stretch")
                
                # Download button
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Predictions",
                    data=csv,
                    file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    width="stretch"
                )
                
                # Show evaluation metrics if labels exist
                if has_labels:
                    st.markdown("---")
                    st.subheader("Evaluation Metrics")
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    
                    # ROC AUC
                    try:
                        y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
                        roc_auc = roc_auc_score(y_true_binarized, y_pred_proba, average='weighted', multi_class='ovr')
                    except:
                        roc_auc = 0.0
                    
                    # Display metrics in columns with better styling
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}", help="Overall correctness of predictions")
                        st.metric("Precision", f"{precision:.4f}", help="Accuracy of positive predictions")
                    
                    with col2:
                        st.metric("Recall", f"{recall:.4f}", help="Coverage of actual positives")
                        st.metric("F1 Score", f"{f1:.4f}", help="Harmonic mean of precision and recall")
                    
                    with col3:
                        st.metric("ROC AUC", f"{roc_auc:.4f}", help="Area under ROC curve")
                        st.metric("MCC", f"{mcc:.4f}", help="Matthews Correlation Coefficient")
                    
                    # Confusion Matrix and Classification Report side by side
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                   xticklabels=sorted(np.unique(y_true)), 
                                   yticklabels=sorted(np.unique(y_true)),
                                   ax=ax, cbar_kws={'label': 'Count'})
                        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
                        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
                        ax.set_title(f'Confusion Matrix\n{selected_model_name}', fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.subheader("Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report).transpose()
                        
                        # Style the dataframe
                        styled_report = report_df.style.background_gradient(
                            cmap='RdYlGn', 
                            subset=['precision', 'recall', 'f1-score'],
                            vmin=0, vmax=1
                        ).format({
                            'precision': '{:.3f}',
                            'recall': '{:.3f}',
                            'f1-score': '{:.3f}',
                            'support': '{:.0f}'
                        })
                        
                        st.dataframe(styled_report, width="stretch")
                    
                    # Prediction distribution
                    st.markdown("---")
                    st.subheader("Prediction Distribution Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        actual_counts = pd.Series(y_true).value_counts().sort_index()
                        colors_actual = ['#3498db', '#2ecc71', '#e74c3c']
                        actual_counts.plot(kind='bar', ax=ax, color=colors_actual[:len(actual_counts)])
                        ax.set_title('Actual Distribution', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Credit Score', fontsize=12)
                        ax.set_ylabel('Count', fontsize=12)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                        
                        # Add value labels on bars
                        for i, v in enumerate(actual_counts):
                            ax.text(i, v + max(actual_counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        pred_counts = pd.Series(y_pred).value_counts().sort_index()
                        colors_pred = ['#3498db', '#2ecc71', '#e74c3c']
                        pred_counts.plot(kind='bar', ax=ax, color=colors_pred[:len(pred_counts)])
                        ax.set_title('Predicted Distribution', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Credit Score', fontsize=12)
                        ax.set_ylabel('Count', fontsize=12)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                        
                        # Add value labels on bars
                        for i, v in enumerate(pred_counts):
                            ax.text(i, v + max(pred_counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig)
                        plt.close()
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                with st.expander("Show error details"):
                    st.exception(e)
    
    else:
        st.info("Click the Predict & Evaluate button in the sidebar to start predictions")

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2 style='color: #0066cc;'>Getting Started</h2>
        <p style='font-size: 18px; color: #333;'>Follow these steps to make predictions:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='step-card'>
            <h3>Step 1: Select Model</h3>
            <p>Choose from 6 trained machine learning models in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='step-card'>
            <h3>Step 2: Upload CSV</h3>
            <p>Upload your test dataset using the file uploader</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='step-card'>
            <h3>Step 3: Predict</h3>
            <p>Click the predict button to get results and metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("Available Models", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            - Logistic Regression
            - Decision Tree
            - K-Nearest Neighbors
            """)
        with cols[1]:
            st.markdown("""
            - Naive Bayes (Gaussian)
            - Random Forest
            - XGBoost
            """)
    
    with st.expander("Features Included"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - Model selection dropdown
            - CSV file upload
            - Comprehensive evaluation metrics
            """)
        with col2:
            st.markdown("""
            - Confusion matrix visualization
            - Classification report
            - Download predictions option
            """)
    
    with st.expander("Need Help?"):
        st.markdown("""
        **Expected CSV Format:**
        - Your CSV should contain all feature columns used during training
        - If you include a `Credit_Score` column, the app will calculate evaluation metrics
        - Without labels, you'll only see predictions
        
        **Common Issues:**
        - Ensure models are trained first by running `model.ipynb`
        - Check that `saved_models/` directory contains all `.pkl` files
        - Verify your CSV columns match the training data format
        """)