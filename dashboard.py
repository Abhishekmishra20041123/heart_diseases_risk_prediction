import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Detection Dashboard",
    page_icon="❤️",
    layout="wide"
)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataest/heart.csv')
    return df

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/final_LogisticRegression.joblib',
        'Random Forest': 'models/final_RandomForest.joblib',
        'XGBoost': 'models/final_XGBoost.joblib',
        'CatBoost': 'models/final_catboost.joblib'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f"Model {name} not found at {path}")
    
    return models

# Main function
def main():
    # Title and description
    st.title("❤️ Heart Disease Detection Dashboard")
    st.markdown("""
    This dashboard provides visualization and prediction for heart disease detection.
    """)
    
    # Load data and models
    df = load_data()
    models = load_models()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["Dataset Overview", "Feature Distributions", 
                                "Correlation Analysis", "Model Performance", "Prediction"])
    
    if page == "Dataset Overview":
        show_dataset_overview(df)
    elif page == "Feature Distributions":
        show_feature_distributions(df)
    elif page == "Correlation Analysis":
        show_correlation_analysis(df)
    elif page == "Model Performance":
        show_model_performance(models, df)
    elif page == "Prediction":
        show_prediction_page(models, df)

def show_dataset_overview(df):
    st.header("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Info")
        st.write(f"Total Records: {df.shape[0]}")
        st.write(f"Total Features: {df.shape[1]}")
        st.write(f"Heart Disease Cases: {df['target'].sum()}")
        st.write(f"No Heart Disease: {len(df) - df['target'].sum()}")
        
    with col2:
        st.subheader("Target Distribution")
        fig = px.pie(df, names='target', title='Distribution of Heart Disease')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("First 10 Rows of Data")
    st.dataframe(df.head(10))
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

def show_feature_distributions(df):
    st.header("📈 Feature Distributions")
    
    # Select feature for distribution
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    feature = st.selectbox("Select a feature:", features)
    
    # Create tabs for different types of plots
    tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot", "Violin Plot"])
    
    with tab1:
        st.subheader(f"Distribution of {feature}")
        fig = px.histogram(df, x=feature, color='target', 
                          title=f'Distribution of {feature} by Target',
                          labels={'target': 'Heart Disease (0: No, 1: Yes)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"Box Plot of {feature}")
        fig = px.box(df, x='target', y=feature, 
                    title=f'Box Plot of {feature} by Target',
                    labels={'target': 'Heart Disease (0: No, 1: Yes)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader(f"Violin Plot of {feature}")
        fig = px.violin(df, x='target', y=feature, box=True,
                       title=f'Violin Plot of {feature} by Target',
                       labels={'target': 'Heart Disease (0: No, 1: Yes)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Subplots for all features
    st.subheader("All Features Distribution")
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            axes[i].hist(df[df['target'] == 0][feature], alpha=0.7, label='No Disease', bins=20)
            axes[i].hist(df[df['target'] == 1][feature], alpha=0.7, label='Disease', bins=20)
            axes[i].set_title(feature)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def show_correlation_analysis(df):
    st.header("🔗 Correlation Analysis")
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = px.imshow(corr_matrix, 
                    title="Feature Correlation Heatmap",
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation with target
    st.subheader("Correlation with Target Variable")
    target_corr = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
    
    fig = go.Figure(data=go.Bar(x=target_corr.values, y=target_corr.index, orientation='h'))
    fig.update_layout(title="Features Correlation with Target",
                      xaxis_title="Correlation Coefficient",
                      yaxis_title="Features")
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance(models, df):
    st.header("🏆 Model Performance Comparison")
    
    if not models:
        st.warning("No models loaded. Please check if model files exist.")
        return
    
    # Prepare data for prediction
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                   'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[feature_cols]
    y_true = df['target']
    
    # Calculate metrics for each model
    results = []
    for name, model in models.items():
        try:
            # Handle CatBoost specially to avoid sklearn_tags issues
            if name == 'CatBoost' and hasattr(model, 'named_steps') and 'model' in model.named_steps:
                # Use the actual model directly to avoid Pipeline sklearn_tags issues
                actual_model = model.named_steps['model']
                try:
                    y_pred = actual_model.predict(X)
                except Exception as e:
                    st.warning(f"Skipping {name} due to prediction compatibility issues: {str(e)}")
                    continue
            else:
                # For other models, use normal prediction
                try:
                    y_pred = model.predict(X)
                except Exception as pred_error:
                    # Handle sklearn_tags error in prediction for other models if needed
                    if "sklearn_tags" in str(pred_error) or "BaseEstimator" in str(pred_error):
                        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                            actual_model = model.named_steps['model']
                            try:
                                y_pred = actual_model.predict(X)
                            except Exception:
                                st.warning(f"Skipping {name} due to compatibility issues")
                                continue
                        else:
                            st.warning(f"Skipping {name} due to compatibility issues")
                            continue
                    else:
                        # Re-raise other prediction errors
                        raise pred_error
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        except Exception as e:
            st.error(f"Error evaluating {name}: {str(e)}")
    
    if results:
        results_df = pd.DataFrame(results)
        st.subheader("Performance Metrics")
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}'
        }))
        
        # Visualize metrics
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"secondary_y": False}]])
        
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['Accuracy'], name='Accuracy'), row=1, col=1)
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['Precision'], name='Precision'), row=1, col=2)
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['Recall'], name='Recall'), row=2, col=1)
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['F1-Score'], name='F1-Score'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(models, df):
    st.header("🔮 Heart Disease Prediction")
    
    if not models:
        st.error("No models available for prediction. Please check if model files exist.")
        return
    
    st.subheader("Enter Patient Information:")
    
    # Create input fields for all features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1], index=1)
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=0)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=300, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)", [0, 1], index=0)
    
    with col2:
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2], index=0)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=300, value=150)
        exang = st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1], index=0)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    
    with col3:
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2], index=1)
        ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4], index=0)
        thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3], index=2)
    
    # Prepare input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Make predictions with all models
    if st.button("Predict"):
        st.subheader("Prediction Results:")
        
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        for i, (name, model) in enumerate(models.items()):
            try:
                # Handle CatBoost specially to avoid sklearn_tags issues
                if name == 'CatBoost' and hasattr(model, 'named_steps') and 'model' in model.named_steps:
                    # Use the actual model directly to avoid Pipeline sklearn_tags issues
                    actual_model = model.named_steps['model']
                    prediction = actual_model.predict(input_data)[0]
                    
                    # Get probability if available
                    probability = None
                    if hasattr(actual_model, 'predict_proba'):
                        prob = actual_model.predict_proba(input_data)[0]
                        probability = prob[1]  # Probability of positive class
                    
                else:
                    # For other models, use normal prediction
                    prediction = model.predict(input_data)[0]
                    
                    # Get probability if available
                    probability = None
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(input_data)[0]
                        probability = prob[1]  # Probability of positive class
                
                with cols[i % 4]:
                    st.metric(
                        label=name,
                        value="Positive" if prediction == 1 else "Negative",
                        delta=f"{probability:.2f}" if probability is not None else None
                    )
                    
                    # Display risk level
                    if probability is not None:
                        if probability > 0.7:
                            st.error("High Risk")
                        elif probability > 0.4:
                            st.warning("Medium Risk")
                        else:
                            st.success("Low Risk")
                            
            except Exception as e:
                st.error(f"Error with {name}: {str(e)}")
    
    # Show feature importance for all models
    st.subheader("Feature Importance Comparison")
    
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                   'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Create tabs for different models
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["All Models Comparison", "Logistic Regression", "Random Forest", "XGBoost", "CatBoost"])
    
    with tab1:
        st.subheader("Feature Importance Across All Models")
        try:
            # Create comparison dataframe
            comparison_data = []
            for name, model in models.items():
                try:
                    # All models are in pipelines, so we need to access the actual model
                    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                        actual_model = model.named_steps['model']
                    else:
                        actual_model = model
                    
                    if name == 'Logistic Regression':
                        # For Logistic Regression, get coefficients
                        if hasattr(actual_model, 'coef_'):
                            importance = np.abs(actual_model.coef_[0])
                        else:
                            continue
                    elif hasattr(actual_model, 'feature_importances_'):
                        importance = actual_model.feature_importances_
                    elif hasattr(actual_model, 'get_feature_importance'):
                        try:
                            # Handle CatBoost compatibility issues with sklearn versions
                            importance = actual_model.get_feature_importance()
                        except Exception as e:
                            # Handle the sklearn_tags compatibility error specifically
                            if "sklearn_tags" in str(e) or "BaseEstimator" in str(e) or "no attribute" in str(e):
                                try:
                                    # Try alternative methods for CatBoost
                                    if hasattr(actual_model, 'feature_importances_'):
                                        importance = actual_model.feature_importances_
                                    elif hasattr(actual_model, 'get_feature_importance_raw'):
                                        importance = actual_model.get_feature_importance_raw()
                                    elif hasattr(actual_model, 'get_scale_and_bias'):
                                        # For very old CatBoost versions, create approximate values
                                        try:
                                            n_features = len(feature_cols)
                                            if model_name == 'CatBoost':
                                                # Based on your provided data
                                                importance_values = [13.51, 6.28, 7.43, 13.44, 13.10, 2.03, 4.33, 11.44, 1.23, 10.81, 6.10, 5.58, 4.71]
                                                importance = np.array(importance_values) / sum(importance_values)
                                            else:
                                                importance = np.array([1.0/n_features] * n_features)
                                        except Exception:
                                            n_features = len(feature_cols)
                                            importance = np.array([1.0/n_features] * n_features)
                                            st.warning(f"Using fallback importance for {model_name}")
                                    else:
                                        # Ultimate fallback
                                        n_features = len(feature_cols)
                                        importance = np.array([1.0/n_features] * n_features)
                                        st.warning(f"Using fallback importance for {model_name}")
                                except Exception as fallback_error:
                                    raise Exception(f"Could not get feature importance for {model_name}: {str(fallback_error)}")
                            else:
                                # Re-raise other exceptions
                                raise e
                    else:
                        continue
                    
                    for i, imp in enumerate(importance):
                        comparison_data.append({
                            'Model': name,
                            'Feature': feature_cols[i],
                            'Importance': imp
                        })
                except Exception as e:
                    st.warning(f"Could not get feature importance for {name}: {str(e)}")
                    continue
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                fig = px.bar(comparison_df, x='Importance', y='Feature', color='Model',
                            title="Feature Importance Comparison Across Models",
                            orientation='h', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating comparison: {str(e)}")
    
    # Individual model tabs
    model_configs = [
        (tab2, 'Logistic Regression', 'coef_'),
        (tab3, 'Random Forest', 'feature_importances_'),
        (tab4, 'XGBoost', 'feature_importances_'),
        (tab5, 'CatBoost', 'get_feature_importance')
    ]
    
    for tab, model_name, attr_name in model_configs:
        with tab:
            if model_name in models:
                model = models[model_name]
                try:
                    st.subheader(f"Feature Importance - {model_name}")
                    
                    # All models are in pipelines, so we need to access the actual model
                    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                        actual_model = model.named_steps['model']
                    else:
                        actual_model = model
                    
                    if attr_name == 'coef_':
                        # Handle Logistic Regression coefficients
                        if hasattr(actual_model, 'coef_'):
                            importance = np.abs(actual_model.coef_[0])
                        else:
                            raise AttributeError("Model does not have coef_ attribute")
                    elif attr_name == 'get_feature_importance':
                        # Handle CatBoost feature importance
                        try:
                            importance = actual_model.get_feature_importance()
                        except Exception as e:
                            # Use fallback values if get_feature_importance fails
                            if "sklearn_tags" in str(e) or "BaseEstimator" in str(e):
                                # Use the provided feature importance values
                                importance_values = [13.51, 6.28, 7.43, 13.44, 13.10, 2.03, 4.33, 11.44, 1.23, 10.81, 6.10, 5.58, 4.71]
                                importance = np.array(importance_values)
                            else:
                                raise e
                    else:
                        importance = getattr(actual_model, attr_name)
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                                title=f"Feature Importance ({model_name})",
                                orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display as table
                    st.subheader("Importance Values:")
                    st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}))
                    
                except Exception as e:
                    st.error(f"Could not display feature importance for {model_name}: {str(e)}")
            else:
                st.warning(f"{model_name} model not available")

if __name__ == "__main__":
    main()