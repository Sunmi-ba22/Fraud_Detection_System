"""
Fraud Detection System - Streamlit Web Application
Author: sunmi_in_tech
Date: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="Locked",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .fraud-warning {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .safe-transaction {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load all saved model artifacts"""
    models_dir = 'models'
    
    try:
        with open(os.path.join(models_dir, 'fraud_detection_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
            feature_columns = pickle.load(f)
        
        with open(os.path.join(models_dir, 'optimal_threshold.pkl'), 'rb') as f:
            threshold = pickle.load(f)
        
        with open(os.path.join(models_dir, 'model_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        return model, scaler, feature_columns, threshold, metadata
    
    except FileNotFoundError:
        st.error(" Model files not found! Please ensure the model has been trained and saved.")
        st.stop()

# Feature engineering function
def engineer_features(transaction_data):
    """Apply the same feature engineering as in training"""
    
    df = transaction_data.copy()
    
    # 1. Amount-based features
    if 'amount' in df.columns:
        df['log_amount'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    
    # 2. Origin balance features
    if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_change_ratio_orig'] = df['balance_change_orig'] / (df['oldbalanceOrg'] + 1)
        df['account_emptied'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
        
        if 'amount' in df.columns:
            df['balance_error_orig'] = abs(df['balance_change_orig'] + df['amount'])
    
    # 3. Destination balance features
    if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['balance_change_ratio_dest'] = df['balance_change_dest'] / (df['oldbalanceDest'] + 1)
        df['dest_account_new'] = (df['oldbalanceDest'] == 0).astype(int)
        
        if 'amount' in df.columns:
            df['balance_error_dest'] = abs(df['balance_change_dest'] - df['amount'])
    
    # 4. Transaction type features
    if 'type' in df.columns:
        type_dummies = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_dummies], axis=1)
        
        df['is_high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    
    # 5. Time-based features
    if 'step' in df.columns:
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = (df['step'] // 24) % 30
        df['is_night'] = ((df['hour_of_day'] >= 3) & (df['hour_of_day'] <= 6)).astype(int)
    
    # 6. Combined features
    if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
        df['amount_pct_of_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_exceeds_balance'] = (df['amount'] > df['oldbalanceOrg']).astype(int)
    
    # Replace infinity with 0
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

# Prediction function
def predict_fraud(transaction_data, model, scaler, feature_columns, threshold):
    """Make fraud prediction on new transaction"""
    
    # Engineer features
    df_engineered = engineer_features(transaction_data)
    
    # Ensure all required features present
    for col in feature_columns:
        if col not in df_engineered.columns:
            df_engineered[col] = 0
    
    # Select and order features
    df_final = df_engineered[feature_columns]
    
    # Scale features
    df_scaled = scaler.transform(df_final)
    
    # Make prediction
    fraud_probability = model.predict_proba(df_scaled)[0, 1]
    is_fraud = int(fraud_probability >= threshold)
    
    # Determine confidence and risk level
    if fraud_probability > 0.8 or fraud_probability < 0.2:
        confidence = 'HIGH'
    elif fraud_probability > 0.6 or fraud_probability < 0.4:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    if fraud_probability > 0.7:
        risk_level = 'HIGH'
    elif fraud_probability > 0.4:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return {
        'is_fraud': is_fraud,
        'fraud_probability': fraud_probability,
        'confidence': confidence,
        'risk_level': risk_level,
        'threshold': threshold
    }

# Main app
def main():
    # Load model
    model, scaler, feature_columns, threshold, metadata = load_model_artifacts()
    
    # Header
    st.title(" Bank Transaction Fraud Detection System")
    st.markdown("---")
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("ℹ Model Information")
        st.write(f"**Model Type:** {metadata.get('model_type', 'N/A')}")
        st.write(f"**Training Date:** {metadata.get('training_date', 'N/A')}")
        st.write(f"**Features:** {metadata.get('n_features', 'N/A')}")
        
        st.markdown("---")
        st.header(" Model Performance")
        metrics = metadata.get('performance_metrics', {})
        st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        st.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        st.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        st.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.2f}%")
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        
        st.markdown("---")
        st.info(" **Tip:** Fill in all transaction details and click 'Analyze Transaction' to check for fraud.")
    
    # Main content - Input form
    st.header(" Enter Transaction Details")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Information")
        
        transaction_type = st.selectbox(
            "Transaction Type",
            options=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],
            help="Type of transaction being performed"
        )
        
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            help="Amount of money being transferred"
        )
        
        step = st.number_input(
            "Time Step (Hour)",
            min_value=0,
            max_value=743,
            value=100,
            help="Hour when transaction occurred (0-743 representing 31 days)"
        )
    
    with col2:
        st.subheader("Origin Account (Sender)")
        
        old_balance_org = st.number_input(
            "Origin Balance Before ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=50000.0,
            step=1000.0,
            help="Account balance before the transaction"
        )
        
        new_balance_orig = st.number_input(
            "Origin Balance After ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=40000.0,
            step=1000.0,
            help="Account balance after the transaction"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Destination Account (Receiver)")
        
        old_balance_dest = st.number_input(
            "Destination Balance Before ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=0.0,
            step=1000.0,
            help="Receiver's account balance before transaction"
        )
        
        new_balance_dest = st.number_input(
            "Destination Balance After ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            help="Receiver's account balance after transaction"
        )
    
    with col4:
        st.subheader("Quick Profiles")
        st.write("Load pre-configured transaction profiles:")
        
        if st.button(" Legitimate Transaction"):
            st.session_state.profile = 'legitimate'
        
        if st.button(" Suspicious Transaction"):
            st.session_state.profile = 'suspicious'
        
        if st.button(" High-Risk Fraud"):
            st.session_state.profile = 'fraud'
    
    # Apply profiles if selected
    if 'profile' in st.session_state:
        if st.session_state.profile == 'legitimate':
            transaction_type = 'PAYMENT'
            amount = 5000.0
            old_balance_org = 50000.0
            new_balance_orig = 45000.0
            old_balance_dest = 10000.0
            new_balance_dest = 15000.0
            st.success(" Loaded: Legitimate Payment Transaction")
        
        elif st.session_state.profile == 'suspicious':
            transaction_type = 'TRANSFER'
            amount = 500000.0
            old_balance_org = 600000.0
            new_balance_orig = 100000.0
            old_balance_dest = 0.0
            new_balance_dest = 500000.0
            st.warning(" Loaded: Suspicious Large Transfer")
        
        elif st.session_state.profile == 'fraud':
            transaction_type = 'CASH_OUT'
            amount = 1000000.0
            old_balance_org = 1000000.0
            new_balance_orig = 0.0
            old_balance_dest = 0.0
            new_balance_dest = 0.0
            st.error(" Loaded: High-Risk Fraud Pattern")
        
        del st.session_state.profile
    
    st.markdown("---")
    
    # Analyze button
    if st.button(" Analyze Transaction", type="primary", use_container_width=True):
        
        # Create transaction dictionary
        transaction = pd.DataFrame([{
            'type': transaction_type,
            'amount': amount,
            'step': step,
            'oldbalanceOrg': old_balance_org,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest
        }])
        
        # Make prediction
        with st.spinner("Analyzing transaction..."):
            result = predict_fraud(transaction, model, scaler, feature_columns, threshold)
        
        # Display results
        st.markdown("---")
        st.header(" Analysis Results")
        
        # Main prediction result
        if result['is_fraud']:
            st.markdown(
                '<div class="fraud-warning"> FRAUD DETECTED - BLOCK TRANSACTION </div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="safe-transaction"> LEGITIMATE TRANSACTION - APPROVE </div>',
                unsafe_allow_html=True
            )
        
        st.markdown("")
        
        # Detailed metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric(
                "Fraud Probability",
                f"{result['fraud_probability']*100:.2f}%",
                delta=f"{(result['fraud_probability'] - threshold)*100:.2f}% vs threshold"
            )
        
        with col_b:
            st.metric("Risk Level", result['risk_level'])
        
        with col_c:
            st.metric("Confidence", result['confidence'])
        
        with col_d:
            st.metric("Decision Threshold", f"{result['threshold']*100:.2f}%")
        
        # Probability gauge
        st.markdown("---")
        st.subheader("Fraud Probability Gauge")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['fraud_probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability (%)", 'font': {'size': 24}},
            delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#00C851'},
                    {'range': [40, 70], 'color': '#ffbb33'},
                    {'range': [70, 100], 'color': '#ff4444'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors
        st.markdown("---")
        st.subheader(" Risk Factor Analysis")
        
        risk_factors = []
        
        # Check various risk factors
        if transaction_type in ['TRANSFER', 'CASH_OUT']:
            risk_factors.append(" High-risk transaction type (TRANSFER/CASH_OUT)")
        
        if amount > 500000:
            risk_factors.append(" Large transaction amount (>$500,000)")
        
        if amount % 100000 == 0:
            risk_factors.append(" Round number amount (potential red flag)")
        
        if new_balance_orig == 0 and old_balance_org > 0:
            risk_factors.append(" Origin account completely emptied")
        
        if old_balance_dest == 0:
            risk_factors.append(" Destination account was previously empty (new account)")
        
        balance_error = abs((new_balance_orig - old_balance_org) + amount)
        if balance_error > 1:
            risk_factors.append(f" Balance inconsistency detected (error: ${balance_error:,.2f})")
        
        if amount > old_balance_org:
            risk_factors.append(" Transaction amount exceeds origin balance")
        
        hour = step % 24
        if 3 <= hour <= 6:
            risk_factors.append(" Transaction during suspicious hours (3 AM - 6 AM)")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success(" No major risk factors detected")
        
        # Recommendation
        st.markdown("---")
        st.subheader(" Recommended Action")
        
        if result['is_fraud']:
            if result['fraud_probability'] > 0.9:
                st.error("**IMMEDIATE ACTION REQUIRED:**\n\n"
                        "1.  Block transaction immediately\n"
                        "2.  Freeze both accounts\n"
                        "3.  Contact account holder for verification\n"
                        "4.  Escalate to fraud investigation team")
            else:
                st.warning("**ENHANCED VERIFICATION REQUIRED:**\n\n"
                          "1. Hold transaction for review\n"
                          "2. Request additional authentication\n"
                          "3. Verify with account holder\n"
                          "4. Manual review by fraud analyst")
        else:
            if result['fraud_probability'] > 0.3:
                st.info("**STANDARD PROCESSING WITH MONITORING:**\n\n"
                       "1. Approve transaction\n"
                       "2. Monitor account for unusual activity\n"
                       "3. Log transaction for pattern analysis")
            else:
                st.success("**APPROVE TRANSACTION:**\n\n"
                          "1.  Process transaction normally\n"
                          "2.  Standard monitoring applies")
        
        # Transaction summary
        st.markdown("---")
        st.subheader("Transaction Summary")
        
        summary_df = pd.DataFrame({
            'Parameter': [
                'Transaction Type',
                'Amount',
                'Origin Balance (Before)',
                'Origin Balance (After)',
                'Destination Balance (Before)',
                'Destination Balance (After)',
                'Time (Hour)',
                'Balance Change (Origin)',
                'Balance Change (Destination)'
            ],
            'Value': [
                transaction_type,
                f"${amount:,.2f}",
                f"${old_balance_org:,.2f}",
                f"${new_balance_orig:,.2f}",
                f"${old_balance_dest:,.2f}",
                f"${new_balance_dest:,.2f}",
                f"{step} (Day {step//24 + 1}, Hour {step%24})",
                f"${new_balance_orig - old_balance_org:,.2f}",
                f"${new_balance_dest - old_balance_dest:,.2f}"
            ]
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()