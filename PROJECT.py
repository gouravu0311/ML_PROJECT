import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import io

# --- Page Config ---
st.set_page_config(page_title="AutoML Pipeline", layout="wide", initial_sidebar_state="expanded")

st.title(" Advanced ML Pipeline Dashboard")
st.markdown("---")

# --- Step 1: Problem Type ---
with st.sidebar:
    st.header("1. Configuration")
    problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Create Horizontal Steps using Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Data & PCA", "🔍 EDA", "🛠️ Engineering", "🎯 Selection", "🤖 Training", "📈 Tuning"
    ])

    # --- Step 2: Input Data & PCA ---
    with t1:
        st.subheader("Data Overview")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            target_col = st.selectbox("Select Target Feature", df.columns)
            feature_cols = st.multiselect("Select Input Features", 
                                          [c for c in df.columns if c != target_col],
                                          default=[c for c in df.columns if c != target_col][:5])
        
        if feature_cols:
            process_df = df[feature_cols].copy()
            # Simple encoding for PCA
            for col in process_df.select_dtypes(include=['object']).columns:
                process_df[col] = LabelEncoder().fit_transform(process_df[col].astype(str))
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(process_df.fillna(0))
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled_data)
            
            fig_pca = px.scatter(components, x=0, y=1, color=df[target_col].astype(str),
                                 title="Data Shape (PCA Projection)",
                                 labels={'0': 'PC1', '1': 'PC2'},
                                 template="plotly_dark")
            st.plotly_chart(fig_pca, use_container_width=True)

    # --- Step 3: EDA ---
    with t2:
        st.subheader("Exploratory Data Analysis")
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.write("Statistics", df.describe())
        with col_eda2:
            st.write("Missing Values", df.isnull().sum())
        
        fig_corr = px.imshow(df.select_dtypes(include=[np.number]).corr(), 
                             text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Step 4: Data Engineering & Cleaning ---
    with t3:
        st.subheader("Outlier Detection & Imputation")
        method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode"])
        outlier_method = st.selectbox("Outlier Detection", ["IQR", "Isolation Forest", "DBSCAN"])
        
        # Simple numeric cleaning for demo
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        
        if outlier_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.1)
            outliers = iso.fit_predict(numeric_df.fillna(0))
            outlier_indices = np.where(outliers == -1)[0]
            st.warning(f"Detected {len(outlier_indices)} outliers.")
            
            if st.button("Remove Outliers"):
                df = df.drop(df.index[outlier_indices])
                st.success("Outliers removed!")

    # --- Step 5: Feature Selection ---
    with t4:
        st.subheader("Feature Importance")
        fs_method = st.selectbox("Selection Method", ["Correlation", "Information Gain", "Variance Threshold"])
        
        if fs_method == "Correlation":
            corrs = df.select_dtypes(include=[np.number]).corr()[target_col].sort_values(ascending=False)
            st.bar_chart(corrs)

    # --- Step 6: Data Split & Model Selection ---
    with t5:
        st.subheader("Model Training")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            k_val = st.number_input("K-Fold Value", min_value=2, max_value=10, value=5)
            
        with col_m2:
            if problem_type == "Regression":
                model_choice = st.selectbox("Model", ["Linear Regression", "SVR", "Random Forest"])
            else:
                model_choice = st.selectbox("Model", ["Logistic Regression", "SVC", "Random Forest"])

        # Dummy preprocessing for training
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        if problem_type == "Classification":
            y = LabelEncoder().fit_transform(y.astype(str))
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Train Model
        if st.button("Train & Validate"):
            if model_choice == "Random Forest":
                model = RandomForestRegressor() if problem_type == "Regression" else RandomForestClassifier()
            elif model_choice == "SVR" or model_choice == "SVC":
                model = SVR() if problem_type == "Regression" else SVC()
            else:
                model = LinearRegression() if problem_type == "Regression" else LogisticRegression()
            
            # K-Fold
            cv_scores = cross_val_score(model, X_train, y_train, cv=int(k_val))
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            st.metric("CV Avg Score", f"{cv_scores.mean():.4f}")
            st.metric("Training Score", f"{train_score:.4f}")
            st.metric("Testing Score", f"{test_score:.4f}")
            
            if train_score > test_score + 0.15:
                st.error("⚠️ Model might be Overfitting")
            elif train_score < 0.5:
                st.warning("⚠️ Model might be Underfitting")

    # --- Step 7: Hyperparameter Tuning ---
    with t6:
        st.subheader("Grid Search Tuning")
        if model_choice == "Random Forest":
            params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
            search = GridSearchCV(model, params, cv=2)
            if st.button("Run Hyperparameter Tuning"):
                with st.spinner("Tuning..."):
                    search.fit(X_train, y_train)
                    st.write("Best Params:", search.best_params_)
                    st.write("Optimized Score:", search.best_score_)
else:
    st.info("Please upload a CSV file to begin the pipeline.")
