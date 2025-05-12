import streamlit as st
st.set_page_config(
    page_title="PA Approval Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import boto3
import pickle
import io
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
import streamlit.components.v1 as components
import shap
from collections import defaultdict

# Add after other global variables
DEBUG_MODE = False  # Set to True only during local testing

def humanize_feature_name(raw_name):
    raw_name = raw_name.replace("cat__", "").replace("num__", "")
    parts = raw_name.split("_")
    
    # Specific mappings
    if "provider" in parts:
        return f"Provider Specialty = {' '.join(parts[2:])}"
    if "documentation" in parts:
        return f"Documentation = {parts[-1]}"
    if "urgency" in parts:
        return f"Urgency Flag = {parts[-1]}"
    if "region" in parts:
        return f"Region = {parts[-1]}"
    if "procedure" in parts:
        return f"Procedure Code = {parts[-1]}"
    if "diagnosis" in parts:
        return f"Diagnosis Code = {parts[-1]}"
    if "patient" in parts and "age" in parts:
        return "Patient Age"
    if "gender" in parts:
        return f"Gender = {parts[-1]}"
    
    # Fallback
    return raw_name.replace("_", " ").title()

@st.cache_data
def load_shap_background():
    import pandas as pd
    df = pd.read_csv('synthetic_pa_dataset.csv')
    # Remove outcome/target column if present
    for col in ['approval_prediction', 'target', 'label', 'outcome']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def process_cpt_codes():
    """
    Process the CPT codes from cpt4.csv and save them to cpt_codes.csv.
    Cleans the codes by stripping whitespace, converting to uppercase,
    removing duplicates and empty codes.
    """
    # Read the input file
    df = pd.read_csv('cpt4.csv')
    
    # Rename columns to match our expected format
    df = df.rename(columns={
        'com.medigy.persist.reference.type.clincial.CPT.code': 'code',
        'label': 'description'
    })
    
    # Clean the codes
    df['code'] = df['code'].astype(str).str.strip().str.upper()
    
    # Drop duplicates and empty codes
    df = df.dropna(subset=['code'])
    df = df.drop_duplicates(subset=['code'])
    
    # Save to CSV
    df.to_csv('cpt_codes.csv', index=False)
    return df

@st.cache_data
def load_cpt_codes():
    """
    Load CPT codes from the CSV file, generating it from cpt4.csv if needed.
    Returns a set of valid codes for validation.
    """
    process_cpt_codes()  # Always (re)generate the CSV for up-to-date codes
    df = pd.read_csv('cpt_codes.csv', dtype={'code': str})
    valid_codes = set(df['code'].str.strip().str.upper())
    return valid_codes

valid_cpt_codes = load_cpt_codes()

# --- ICD-10 code processing and loading ---
def process_icd10_codes():
    """
    Process the ICD-10 codes from icd10_2025.txt and save them to icd10_codes.csv.
    Each line in the input file should be in the format: 'CODE Description'
    """
    codes_data = []
    with open('icd10_2025.txt', 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                code = parts[0]
                description = parts[1]
                codes_data.append({'code': code, 'description': description})
    df = pd.DataFrame(codes_data)
    df.to_csv('icd10_codes.csv', index=False)
    return df

@st.cache_data
def load_icd10_codes():
    """
    Load ICD-10 codes from the CSV file, generating it from icd10_2025.txt if needed.
    Returns a set of valid codes for validation.
    """
    process_icd10_codes()  # Always (re)generate the CSV for up-to-date codes
    df = pd.read_csv('icd10_codes.csv')
    valid_codes = set(df['code'].str.strip().str.upper())
    return valid_codes

valid_icd10_codes = load_icd10_codes()

@st.cache_data
def load_provider_specialties():
    df = pd.read_csv('provider_specialties.csv')
    valid_specialties = df['specialty'].dropna().tolist()
    return valid_specialties

valid_specialties = load_provider_specialties()


# Helper functions
def load_model_from_s3():
    """
    Always load the latest model from S3 and print/display the S3 LastModified timestamp.
    This model is trained on the expanded set of ICD-10 (74,000+) and CPT (8,000+) codes.
    """
    import boto3
    import pickle
    from datetime import datetime
    BUCKET_NAME = 'pa-predictor-bucket-rs'
    MODEL_FILE = 'pa_predictor_model.pkl'
    s3 = boto3.client('s3')
    try:
        # Get object metadata for timestamp
        obj_metadata = s3.head_object(Bucket=BUCKET_NAME, Key=MODEL_FILE)
        last_modified = obj_metadata['LastModified']
        print(f"Loading model from S3: s3://{BUCKET_NAME}/{MODEL_FILE}")
        print(f"S3 LastModified: {last_modified} (UTC)")
        # Download and load the model
        response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_FILE)
        model = pickle.loads(response['Body'].read())
        # Display timestamp and version info in Streamlit UI
        st.info(f"Model loaded from S3 at: {last_modified} (UTC)")
        st.info("Model Version: Updated with expanded ICD-10 (74,000+) and CPT (8,000+) codes (May 2025)")
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        st.error(f"Error loading model from S3: {e}")
        return None

def call_backend_api(input_dict):
    """
    Call the FastAPI backend to get predictions and feature importance
    """
    try:
        url = "http://54.163.203.207:8001/predict"
        response = requests.post(url, json=input_dict, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.json().get('detail')}")
            return None
    except Exception as e:
        st.error(f"❌ Failed to contact backend: {e}")
        return None

def get_recommendation(docs_complete, prior_denials):
    """Generate recommendation based on input features"""
    if docs_complete == 'N':
        return "Submit missing documentation"
    elif prior_denials > 5:
        return "Consider peer review or alternative procedure code"
    else:
        return "No additional action recommended"

def log_user_prediction(input_data, prediction, probability):
    try:
        # Combine user inputs + results into one row
        row = input_data.copy()
        row['prediction'] = prediction
        row['probability'] = round(probability, 2)
        row['timestamp'] = datetime.utcnow().isoformat()

        df = pd.DataFrame([row])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        # S3 key with timestamp
        key = f"predictions/prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

        s3.put_object(
            Bucket=LOG_BUCKET,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print("✅ Prediction logged to S3:", key)
    except Exception as e:
        print("❌ Logging to S3 failed:", str(e))

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def create_gauge_chart(probability):
    """Create a gauge chart for approval probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={'suffix': '%', 'font': {'size': 28}},  # Increased font size
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#38b2ac"},
            'steps': [
                {'range': [0, 30], 'color': "#fc8181"},
                {'range': [30, 60], 'color': "#f6ad55"},
                {'range': [60, 100], 'color': "#68d391"}  # Adjusted green zone
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60  # Adjusted threshold
            }
        },
        title={'text': ''}
    ))
    fig.update_layout(
        title_text='',
        height=300,
        width=500,  # Increased width
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2c5282", size=14)  # Increased base font size
    )
    return fig

def get_feature_importance(model, input_data, background_data=None):
    """
    Calculate SHAP values locally using the loaded model and return a DataFrame
    with human-readable top features and their impact.
    """
    try:
        # Ensure input is DataFrame
        if not isinstance(input_data, pd.DataFrame):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Use input as background if none provided
        if background_data is None:
            background_data = input_df.copy()

        # SHAP explainer
        explainer = shap.Explainer(
            model.predict_proba,
            background_data,
            algorithm="permutation",
            feature_names=input_df.columns
        )
        shap_values = explainer(input_df)
        shap_vals = shap_values.values[0]

        # Top 3 features by absolute SHAP value
        top_idx = np.argsort(-np.abs(shap_vals))[:3]
        features = []
        for i in top_idx:
            fname = input_df.columns[i]
            fval = input_df.iloc[0, i]
            impact = "positive" if shap_vals[i] > 0 else "negative"
            direction = "↑" if shap_vals[i] > 0 else "↓"
            # Humanize feature name/value
            if "provider_specialty" in fname:
                label = f"Provider Specialty = {fval}"
            elif "documentation_complete" in fname:
                label = f"Documentation Complete = {fval}"
            elif "prior_denials" in fname:
                label = f"Prior Denials = {fval}"
            elif "procedure_code" in fname:
                label = f"Procedure Code = {fval}"
            elif "diagnosis_code" in fname:
                label = f"Diagnosis Code = {fval}"
            else:
                label = f"{fname.replace('_', ' ').title()} = {fval}"
            features.append({
                "Feature": label,
                "Effect": impact,
                "Direction": direction,
                "Importance": abs(shap_vals[i])
            })
        df = pd.DataFrame(features)
        return df, None, shap_vals, None, None
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return pd.DataFrame(), "No explanation available.", None, None, None

def get_local_shap_explanation(input_df, model, preprocessor, classifier, background_df):
    try:
        # Preprocess input and background
        X_input = preprocessor.transform(input_df)
        X_bg = preprocessor.transform(background_df.sample(n=100, random_state=42))

        # Create SHAP explainer
        explainer = shap.KernelExplainer(classifier.predict_proba, X_bg)
        shap_values = explainer.shap_values(X_input, nsamples=100)

        print("✅ SHAP raw type:", type(shap_values))
        print("✅ SHAP raw shape (if list):", [np.array(s).shape for s in shap_values] if isinstance(shap_values, list) else np.array(shap_values).shape)

        try:
            # Extract SHAP values for class 1 (approval)
            shap_vals = shap_values[0, :, 1]  # First sample, all features, class 1 (approval)
            shap_vals = np.array(shap_vals).flatten()
            feature_names = preprocessor.get_feature_names_out()

            # Filter only one-hot features that are "on" (value = 1) in this input
            X_input_flat = X_input[0]  # shape: (n_features,)
            active_indices = [i for i, val in enumerate(X_input_flat) if val == 1]

            # Subset SHAP values and feature names to active only
            shap_vals = shap_vals[active_indices]
            feature_names = [feature_names[i] for i in active_indices]

            print("✅ SHAP computed")
            print("SHAP values:", shap_vals[:5])
            print("Feature names:", feature_names[:5])

            # Top 3 features by impact
            top_idx = np.argsort(-np.abs(shap_vals))[:3]
            explanations = []

            for i in top_idx:
                fname = str(feature_names[i])
                direction = '↑ approval' if shap_vals[i] > 0 else '↓ approval'

                try:
                    if 'provider_specialty' in fname:
                        val = input_df['provider_specialty'].values[0]
                        label = f"Provider Specialty = {val}"
                    elif 'documentation_complete' in fname:
                        val = input_df['documentation_complete'].values[0]
                        label = f"Documentation Complete = {val}"
                    elif 'prior_denials' in fname:
                        val = input_df['prior_denials_provider'].values[0]
                        label = f"Prior Denials = {val}"
                    elif 'procedure_code' in fname:
                        val = input_df['procedure_code'].values[0]
                        label = f"Procedure Code = {val}"
                    elif 'diagnosis_code' in fname:
                        val = input_df['diagnosis_code'].values[0]
                        label = f"Diagnosis Code = {val}"
                    else:
                        label = fname.replace('cat__', '').replace('_', ' ').title()

                    explanations.append(f"{label} ({direction})")
                except Exception as e:
                    print(f"⚠️ Failed to interpret feature '{fname}': {e}")
                    explanations.append(f"{fname} ({direction})")

            # Deduplicate explanations while preserving order
            explanations = list(dict.fromkeys(explanations))

            return explanations, shap_vals, feature_names, shap_values

        except Exception as e:
            print(f"❌ SHAP extraction failed: {e}")
            return ["No explanation available."], None, None, None

    except Exception as e:
        print(f"❌ SHAP explanation error: {e}")
        return ["No explanation available."], None, None, None

@st.cache_resource
def load_model_and_components():
    model = load_model_from_s3()
    if model is not None:
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        return model, preprocessor, classifier
    return None, None, None

# Initialize S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'pa-predictor-bucket-rs'
MODEL_FILE = 'pa_predictor_model.pkl'
LOG_BUCKET = 'pa-predictor-logs'  # create this if it doesn't exist


# Custom CSS styling with Google Fonts
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        /* Global theme variables */
        :root {
            --primary-color: #0ea5e9;
            --primary-hover: #0284c7;
            --secondary-color: #64748b;
            --success-color: #059669;
            --error-color: #dc2626;
            --background-color: #f8fafc;
            --card-background: #ffffff;
            --border-color: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --spacing-xs: 0.5rem;
            --spacing-sm: 0.75rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
        }

        /* Global font settings */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
            line-height: 1.5;
        }
        
        /* Main background color */
        .stApp {
            background-color: var(--background-color);
        }
        
        /* Header styling */
        .header-section {
            background-image: linear-gradient(rgba(14, 165, 233, 0.95), rgba(2, 132, 199, 0.95)), 
                            url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-position: center;
            padding: var(--spacing-xl) var(--spacing-lg);
            margin: -1rem -1rem var(--spacing-xl) -1rem;
            color: white;
            text-align: center;
            border-radius: 0;
            box-shadow: var(--shadow-lg);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: var(--spacing-xs);
            color: white;
            letter-spacing: -0.025em;
        }
        
        .header-subtitle {
            font-size: 1.125rem;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 0;
            font-weight: 400;
        }
        
        /* Card styling */
        .card-container {
            max-width: 700px !important;
            margin: 2.5rem auto 2rem auto !important;
            padding: 2.5rem 2rem 2rem 2rem !important;
            box-shadow: 0 4px 24px 0 rgba(16,30,54,0.10);
            border-radius: 1.25rem;
            border: 1px solid #e2e8f0;
            background: #fff;
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: var(--spacing-lg);
            padding-bottom: var(--spacing-sm);
            border-bottom: 1px solid var(--border-color);
            letter-spacing: -0.025em;
        }
        
        /* Form styling */
        .stSelectbox, .stNumberInput {
            background-color: var(--card-background);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-color);
            margin-bottom: var(--spacing-sm);
        }
        
        /* Form labels */
        .stSelectbox label, .stNumberInput label {
            font-weight: 500;
            font-size: 0.875rem;
            color: var(--text-primary);
            margin-bottom: var(--spacing-xs);
            display: block;
        }
        
        /* Form values - high contrast for accessibility */
        .stSelectbox div[data-baseweb="select"] input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input {
            color: #222 !important;
            font-size: 0.95rem;
            font-weight: 500;
        }
        /* Placeholder text lighter, but not too faint */
        .stSelectbox div[data-baseweb="select"] input::placeholder,
        .stNumberInput input::placeholder {
            color: #888 !important;
            opacity: 1;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            font-weight: 500;
            font-size: 0.875rem;
            border-radius: var(--radius-sm);
            border: none;
            padding: var(--spacing-xs) var(--spacing-md);
            transition: all 0.2s ease;
            height: 2.5rem;
            box-shadow: var(--shadow-sm);
        }
        
        .stButton > button:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        /* Step indicator */
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin: 0 auto var(--spacing-xl);
            padding: 0 var(--spacing-lg);
            max-width: 1000px;
            position: relative;
        }
        
        .step-indicator::before {
            content: '';
            position: absolute;
            top: 1rem;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            height: 2px;
            background-color: var(--border-color);
            z-index: 1;
        }
        
        .step {
            text-align: center;
            flex: 1;
            position: relative;
            z-index: 2;
        }
        
        .step-number {
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            background-color: var(--card-background);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto var(--spacing-xs);
            font-size: 0.875rem;
            font-weight: 500;
            border: 2px solid var(--border-color);
            transition: all 0.2s ease;
        }
        
        .step.active .step-number {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .step-label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }
        
        .step.active .step-label {
            color: var(--primary-color);
        }
        
        /* Results section */
        .results-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--spacing-lg);
            margin-top: var(--spacing-lg);
        }
        
        .result-card {
            background-color: var(--card-background);
            border-radius: var(--radius-md);
            padding: var(--spacing-lg);
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }
        
        /* Status badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--radius-sm);
            font-size: 0.875rem;
            font-weight: 500;
            margin-top: var(--spacing-sm);
        }
        
        .status-badge.approved {
            background-color: #dcfce7;
            color: var(--success-color);
        }
        
        .status-badge.denied {
            background-color: #fee2e2;
            color: var(--error-color);
        }
        
        /* Info box styling */
        .stInfo {
            background-color: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            margin: var(--spacing-md) 0;
            font-size: 0.875rem;
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: #f0fdf4;
            border: 1px solid #86efac;
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            margin: var(--spacing-md) 0;
            font-size: 0.875rem;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: var(--spacing-lg);
            color: var(--text-secondary);
            font-size: 0.875rem;
            border-top: 1px solid var(--border-color);
            margin-top: var(--spacing-xl);
            background-color: var(--card-background);
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .header-title {
                font-size: 2rem;
            }
            
            .card-container {
                padding: var(--spacing-md);
            }
            
            .results-container {
                grid-template-columns: 1fr;
            }
            
            .step-indicator::before {
                display: none;
            }
        }

        .stPlotlyChart {
            padding: 0 !important;
            margin: 0 !important;
        }
        .stPlotlyChart .xtick, .stPlotlyChart .ytick, .stPlotlyChart .legendtext, .stPlotlyChart .title {
            font-family: 'Inter', sans-serif !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #222 !important;
        }
        .divider-vertical {
            width: 1.5px;
            background: #e2e8f0;
            margin: 0 1.5rem;
            height: 220px;
            align-self: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load healthcare animation
# lottie_healthcare = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_1pxqjqps.json")

# Sidebar menu
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=100)
    selected = option_menu(
        menu_title="Navigation",
        options=["Predict", "About"],
        icons=["clipboard2-pulse", "info-circle"],
        menu_icon="hospital",
        default_index=0,
    )

# Main content
if selected == "Predict":
    # Header section
    st.markdown("""
        <div class="header-section">
            <h1 class="header-title">Prior Authorization Predictor</h1>
            <p class="header-subtitle">AI-powered approval prediction for healthcare providers</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for wizard steps and form data
    if 'step' not in st.session_state:
        st.session_state.step = 1
        st.session_state.form_data = {
            'patient_age': None,
            'patient_gender': None,
            'procedure_code': '',
            'diagnosis_code': '',
            'provider_specialty': None,
            'payer': None,
            'urgency_flag': None,
            'documentation_complete': None,
            'prior_denials': 0,
            'region': None
        }
    
    # Step indicator
    st.markdown("""
        <div class="step-indicator">
            <div class="step {}">
                <div class="step-number">1</div>
                <div class="step-label">Patient Information</div>
            </div>
            <div class="step {}">
                <div class="step-number">2</div>
                <div class="step-label">Request Details</div>
            </div>
        </div>
    """.format(
        'active' if st.session_state.step == 1 else '',
        'active' if st.session_state.step == 2 else ''
    ), unsafe_allow_html=True)
    
    # Step 1: Patient Information
    if st.session_state.step == 1:
        st.markdown("""
            <div class="card-container">
                <div class="card-title">Patient Information</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.form_data['patient_age'] = st.number_input(
                "Patient Age", 
                min_value=18, 
                max_value=90, 
                value=st.session_state.form_data['patient_age'] or 18,  # Default to 18 if None
                help="Enter patient age between 18 and 90"
            )
            
            # Updated gender dropdown with placeholder
            gender_options = ['Select Gender', 'M', 'F']
            current_gender = st.session_state.form_data['patient_gender']
            gender_index = 0 if current_gender is None else gender_options.index(current_gender)
            st.session_state.form_data['patient_gender'] = st.selectbox(
                "Patient Gender", 
                options=gender_options,
                index=gender_index
            )
            
            # Add validation message for gender
            if st.session_state.form_data['patient_gender'] == 'Select Gender':
                st.warning("⚠️ Please select a valid gender.")
            
            # Procedure Code input and validation
            st.session_state.form_data['procedure_code'] = st.text_input(
                "Procedure Code (enter CPT code):",
                value=st.session_state.form_data['procedure_code'],
                placeholder="Enter a valid CPT code",
                help="Validated against expanded CPT code set (8,000+ codes)"
            )
            input_proc_code = st.session_state.form_data['procedure_code'].strip().upper()
            if input_proc_code:
                if input_proc_code not in valid_cpt_codes:
                    st.warning(f"⚠️ The procedure code '{input_proc_code}' is not a valid CPT code. Please correct it before proceeding.")
                else:
                    st.success(f"✅ Procedure code '{input_proc_code}' is valid.")
        with col2:
            # Diagnosis Code input and validation
            st.session_state.form_data['diagnosis_code'] = st.text_input(
                "Diagnosis Code",
                value=st.session_state.form_data['diagnosis_code'],
                placeholder="Enter a valid ICD-10 code",
                max_chars=10,
                help="Validated against expanded ICD-10 code set (74,000+ codes)"
            )
            
            # Updated provider specialty dropdown with placeholder
            specialty_options = ['Select Provider Specialty'] + valid_specialties
            current_specialty = st.session_state.form_data['provider_specialty']
            specialty_index = 0 if current_specialty is None else specialty_options.index(current_specialty)
            st.session_state.form_data['provider_specialty'] = st.selectbox(
                "Provider Specialty",
                options=specialty_options,
                index=specialty_index
            )
            
            # Add validation message for provider specialty
            if st.session_state.form_data['provider_specialty'] == 'Select Provider Specialty':
                st.warning("⚠️ Please select a valid provider specialty.")
            
            input_diag_code = st.session_state.form_data['diagnosis_code'].strip().upper()
            if input_diag_code:
                if input_diag_code not in valid_icd10_codes:
                    st.warning(f"⚠️ The diagnosis code '{input_diag_code}' is not a valid ICD-10 code. Please correct it before proceeding.")
                else:
                    st.success(f"✅ Diagnosis code '{input_diag_code}' is valid.")
        
        # Only enable Next Step if all required fields are valid
        codes_valid = (
            input_proc_code in valid_cpt_codes and
            input_diag_code in valid_icd10_codes and
            st.session_state.form_data['patient_gender'] in ['M', 'F'] and
            st.session_state.form_data['provider_specialty'] in valid_specialties
        )
        
        if st.button("Next Step", key="next_step_1"):
            if codes_valid:
                st.session_state.step = 2
                st.rerun()
            else:
                error_messages = []
                if not input_proc_code or input_proc_code not in valid_cpt_codes:
                    error_messages.append("Please enter a valid procedure code")
                if not input_diag_code or input_diag_code not in valid_icd10_codes:
                    error_messages.append("Please enter a valid diagnosis code")
                if st.session_state.form_data['patient_gender'] not in ['M', 'F']:
                    error_messages.append("Please select a valid gender")
                if st.session_state.form_data['provider_specialty'] not in valid_specialties:
                    error_messages.append("Please select a valid provider specialty")
                
                st.error("⚠️ " + "\n".join(error_messages))
    
    # Step 2: Request Details
    elif st.session_state.step == 2:
        st.markdown("""
            <div class="card-container">
                <div class="card-title">Request Details</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if DEBUG_MODE:
                st.write("🔍 Session Keys:", list(st.session_state.keys()))
            try:
                # Load payers dynamically from dataset
                payer_df = pd.read_csv('synthetic_pa_dataset_v2.csv')
                payer_options = sorted(payer_df['payer'].dropna().unique().tolist())
                # Filter out any non-payer values (like header rows)
                payer_options = [p for p in payer_options if p not in ['payer', '']]
                st.write(f"⚠️ Loaded {len(payer_options)} unique insurance payers from dataset.")
                st.session_state.form_data['payer'] = st.selectbox(
                    "Insurance Payer",
                    payer_options,
                    index=payer_options.index(st.session_state.form_data['payer']) if st.session_state.form_data['payer'] in payer_options else 0
                )
                urgency_options = ['Y', 'N']
                current_urgency = st.session_state.form_data['urgency_flag']
                urgency_index = urgency_options.index(current_urgency) if current_urgency in urgency_options else 0
                st.session_state.form_data['urgency_flag'] = st.selectbox(
                    "Urgent Request", 
                    urgency_options,
                    index=urgency_index
                )
            except Exception as e:
                st.error(f"❌ Streamlit Cloud layout error: {e}")
                st.stop()
        with col2:
            doc_options = ['Y', 'N']
            current_doc = st.session_state.form_data['documentation_complete']
            doc_index = doc_options.index(current_doc) if current_doc in doc_options else 0
            st.session_state.form_data['documentation_complete'] = st.selectbox(
                "Documentation Complete", 
                doc_options,
                index=doc_index
            )
            st.session_state.form_data['prior_denials'] = st.number_input(
                "Prior Denials", 
                min_value=0, 
                max_value=10, 
                value=st.session_state.form_data['prior_denials']
            )
            region_options = ['Midwest', 'Northeast', 'South', 'West']
            current_region = st.session_state.form_data['region']
            region_index = region_options.index(current_region) if current_region in region_options else 0
            st.session_state.form_data['region'] = st.selectbox(
                "Region", 
                region_options,
                index=region_index
            )
        input_code = st.session_state.form_data['diagnosis_code'].strip().upper()
        input_proc_code = st.session_state.form_data['procedure_code'].strip().upper()
        if input_code:
            if input_code not in valid_icd10_codes:
                st.warning(f"⚠️ The diagnosis code '{input_code}' is not a valid ICD-10 code. Please correct it before proceeding.")
            else:
                st.success(f"✅ Diagnosis code '{input_code}' is valid.")


        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Back", key="back_step_2"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Predict Approval", key="predict_button"):
                # Block prediction if either code is invalid
                if input_proc_code not in valid_cpt_codes:
                    st.error("Prediction blocked → procedure code is invalid.")
                elif input_code not in valid_icd10_codes:
                    st.error("Prediction blocked → diagnosis code is invalid.")
                else:
                    # Create input DataFrame using session state data
                    input_data = pd.DataFrame({
                        'patient_age': [st.session_state.form_data['patient_age']],
                        'patient_gender': [st.session_state.form_data['patient_gender']],
                        'procedure_code': [st.session_state.form_data['procedure_code']],
                        'diagnosis_code': [st.session_state.form_data['diagnosis_code']],
                        'provider_specialty': [st.session_state.form_data['provider_specialty']],
                        'payer': [st.session_state.form_data['payer']],
                        'urgency_flag': [st.session_state.form_data['urgency_flag']],
                        'documentation_complete': [st.session_state.form_data['documentation_complete']],
                        'prior_denials_provider': [st.session_state.form_data['prior_denials']],
                        'region': [st.session_state.form_data['region']]
                    })

                    # Make prediction
                    prediction_result = call_backend_api(input_data.iloc[0].to_dict())
                    
                    if prediction_result:
                        st.success("✅ Prediction received from API.")
                        
                        # Extract and format values from API response
                        approval = prediction_result.get("approval_prediction", 0)
                        probability = float(prediction_result.get("probability", 0.0))
                        probability_pct = f"{probability * 100:.1f}%"
                        status_str = "Approved" if approval == 1 else "Denied"
                        status_color = "green" if approval == 1 else "red"
                        status_emoji = "✅" if approval == 1 else "❌"

                        # Debug section (only visible in debug mode)
                        if DEBUG_MODE:
                            with st.expander("Advanced Debug View", expanded=False):
                                st.markdown("### Raw Prediction Data")
                                st.json(prediction_result)
                                
                                st.markdown("### Model Metadata")
                                st.info(f"Model Version: Updated with expanded ICD-10 (74,000+) and CPT (8,000+) codes (May 2025)")
                                st.info(f"Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

                        # Display Prediction Summary first
                        st.markdown("### ✅ Prediction Summary")
                        st.markdown("---")
                        
                        # Debug information
                        if DEBUG_MODE:
                            st.write("🔍 Session Keys:", list(st.session_state.keys()))
                        
                        try:
                            # Create columns for status and probability
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.markdown(f"**Approval Status**")
                                st.markdown(f"<span style='color:{status_color}; font-size:1.3em; font-weight:600'>{status_emoji} {status_str}</span>", unsafe_allow_html=True)
                            with col2:
                                st.markdown("**Approval Probability**")
                                st.markdown(f"<span style='font-size:1.3em; font-weight:600'>{probability_pct}</span>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Streamlit Cloud layout error: {e}")
                            # Fallback display without columns
                            st.markdown(f"**Approval Status:** <span style='color:{status_color}; font-size:1.3em; font-weight:600'>{status_emoji} {status_str}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Approval Probability:** <span style='font-size:1.3em; font-weight:600'>{probability_pct}</span>", unsafe_allow_html=True)
                            st.stop()

                        st.markdown("---")
                        
                        # Recommendation logic
                        if approval == 1 and probability < 0.9:
                            recommendation = "Consider strengthening documentation or verifying procedure details."
                        elif approval == 0:
                            recommendation = "Review contributing features and revise submission as needed."
                        else:
                            recommendation = "No additional documentation likely needed."
                        
                        st.markdown(f"💬 **Recommendation:** {recommendation}")
                        st.markdown("---")

                        # Add prediction timestamp
                        st.caption(f"🕒 Prediction generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

                        # SHAP: Use local model to get feature importance
                        model, preprocessor, classifier = load_model_and_components()
                        background_data = load_shap_background()
                        explanations, shap_vals, feature_names, shap_full = get_local_shap_explanation(
                            input_df=input_data,
                            model=model,
                            preprocessor=preprocessor,
                            classifier=classifier,
                            background_df=background_data
                        )

                        # Display Top Contributing Factors with emoji enhancement
                        if explanations and isinstance(explanations, list) and any("no explanation" not in str(exp).lower() for exp in explanations):
                            st.markdown("### 🔬 Top Contributing Factors")
                            for exp in explanations:
                                if "↑" in exp:
                                    icon = "⬆️"
                                elif "↓" in exp:
                                    icon = "⬇️"
                                else:
                                    icon = "🔹"
                                st.markdown(f"{icon} **{exp}**")
                        else:
                            st.markdown("### 🔬 Top Contributing Factors")
                            st.markdown("No explanation available.")

                        # Add SHAP bar chart visualization
                        if shap_vals is not None and feature_names is not None:
                            # Add chart interpretation guide
                            st.markdown("""
                            **How to interpret this chart:**
                            The bar chart below shows which features had the biggest impact on the approval prediction for this request. 
                            - Blue bars = features that increased the likelihood of approval.
                            - Red bars = features that decreased it.
                            Longer bars indicate stronger influence.
                            """)

                            # Humanize feature names
                            human_names = [humanize_feature_name(name) for name in feature_names]
                            shap_df = pd.DataFrame({
                                "Feature": human_names,
                                "SHAP Value": shap_vals
                            })
                            shap_df["abs_val"] = shap_df["SHAP Value"].abs()
                            shap_df = shap_df.sort_values(by="abs_val", ascending=False).head(5)

                            fig = px.bar(
                                shap_df, 
                                x="SHAP Value", 
                                y="Feature", 
                                orientation="h",
                                title="Top SHAP Feature Contributions",
                                color="SHAP Value",
                                color_continuous_scale="RdBu",
                                template="plotly_white",
                                height=300
                            )
                            fig.update_layout(
                                margin=dict(l=10, r=10, t=40, b=10),
                                xaxis_title="Impact on Approval Likelihood",
                                yaxis_title=None,
                                font=dict(size=14),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        st.markdown("---")

    # Footer
    st.markdown("""
        <div class="footer">
            © 2025 PA Predictor. For demonstration purposes only.<br>
            Contact: ravikirans723@gmail.com
        </div>
    """, unsafe_allow_html=True)

    # Add version label at the bottom of the app
    st.markdown("""
        <div style='position: fixed; bottom: 0; width: 100%; background-color: #f0f2f6; padding: 10px; text-align: center;'>
            <p style='margin: 0; color: #666; font-size: 0.8em;'>
                Model Version: Updated with expanded ICD-10 (74,000+) and CPT (8,000+) codes (May 2025)
            </p>
        </div>
    """, unsafe_allow_html=True)

elif selected == "About":
    st.markdown('<h1 class="custom-title">About the PA Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    ### What is Prior Authorization?
    Prior Authorization (PA) is a process used by health insurance companies to determine if they will cover a prescribed procedure, service, or medication.
    
    ### How It Works
    This tool uses machine learning to predict the likelihood of PA approval based on various factors including:
    - Patient demographics
    - Procedure and diagnosis codes
    - Provider specialty
    - Insurance payer
    - Request urgency
    - Documentation status
    
    ### Model Information
    - Trained on 10,000+ historical PA requests
    - Achieves 85%+ accuracy
    - Continuously updated with new data
    
    ### Contact
    For support or questions, please contact the development team.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model_locally_from_s3():
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket="pa-predictor-bucket-rs", Key="pa_predictor_model.pkl")
    model = pickle.loads(response['Body'].read())
    return model

# Load the local model for SHAP explanations
local_model = load_model_locally_from_s3()

