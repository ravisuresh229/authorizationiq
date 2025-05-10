import streamlit as st
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

st.set_page_config(
    page_title="PA Approval Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_cpt_codes():
    df = pd.read_csv('cpt_codes.csv', dtype={'code': str})
    valid_codes = set(df['code'].str.strip().str.upper())
    return valid_codes

valid_cpt_codes = load_cpt_codes()



@st.cache_data
def load_icd10_codes():
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
        # Display timestamp in Streamlit UI
        st.info(f"Model loaded from S3 at: {last_modified} (UTC)")
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        st.error(f"Error loading model from S3: {e}")
        return None

def get_recommendation(docs_complete, prior_denials):
    """Generate recommendation based on input features"""
    if docs_complete == 'N':
        return "Submit missing documentation"
    elif prior_denials > 5:
        return "Consider peer review or alternative procedure code"
    else:
        return "No additional action recommended"

def log_prediction(input_data, prediction, probability):
    """Log the prediction to S3 (optional)"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            'timestamp': timestamp,
            'input_data': input_data,
            'prediction': prediction,
            'probability': probability
        }
        
        # Convert to CSV format
        log_df = pd.DataFrame([log_data])
        csv_buffer = io.StringIO()
        log_df.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f'predictions/prediction_{timestamp}.csv',
            Body=csv_buffer.getvalue()
        )
        return True
    except Exception as e:
        st.warning(f"Could not log prediction: {e}")
        return False

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
        number={'suffix': '%'},  # Add percentage sign
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#38b2ac"},
            'steps': [
                {'range': [0, 30], 'color': "#fc8181"},
                {'range': [30, 70], 'color': "#f6ad55"},
                {'range': [70, 100], 'color': "#68d391"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        },
        title={'text': ''}
    ))
    fig.update_layout(
        title_text='',
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2c5282")
    )
    return fig

def get_feature_importance(model, input_data, background_data=None):
    """
    Get feature importance for the prediction using SHAP.
    Returns:
    - feature_importance_df: DataFrame with top 5 feature importances
    - explanation: Narrative explanation of the top feature
    - original_shap: SHAP value of the top feature
    - full_feature_importance_df: Filtered DataFrame with all meaningful feature importances
    - grouped_importance_df: DataFrame with features grouped by category
    """
    # Use background_data for SHAP if provided, else use input_data
    if background_data is None:
        background_data = input_data

    # Get the preprocessor and inspect its type
    preprocessor = model.named_steps['preprocessor']
    print(f"Preprocessor type: {type(preprocessor)}")
    
    # Get preprocessed input first to know the expected number of features
    preprocessed_input = preprocessor.transform(input_data)
    if hasattr(preprocessed_input, 'toarray'):
        preprocessed_input = preprocessed_input.toarray()
    print(f"Preprocessed input shape: {preprocessed_input.shape}")

    # Get feature names based on preprocessor type
    if hasattr(preprocessor, 'get_feature_names_out'):
        # For ColumnTransformer and other modern sklearn transformers
        feature_names = preprocessor.get_feature_names_out()
        print(f"Number of feature names from get_feature_names_out(): {len(feature_names)}")
        
        # If we still have a mismatch, try to get names from individual transformers
        if len(feature_names) != preprocessed_input.shape[1]:
            print("Mismatch detected, trying to get names from individual transformers...")
            if hasattr(preprocessor, 'transformers_'):
                # For ColumnTransformer, collect names from each transformer
                all_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if hasattr(trans, 'get_feature_names_out'):
                        trans_names = trans.get_feature_names_out(cols)
                        all_names.extend(trans_names)
                    else:
                        # For transformers without get_feature_names_out, use column names
                        all_names.extend(cols)
                feature_names = np.array(all_names)
                print(f"Number of feature names from transformers: {len(feature_names)}")
    else:
        # Fallback: use input column names
        feature_names = input_data.columns
        print(f"Using input column names: {len(feature_names)}")

    # Ensure feature_names matches the preprocessed input shape
    if len(feature_names) != preprocessed_input.shape[1]:
        print(f"Warning: Feature name count ({len(feature_names)}) doesn't match preprocessed input columns ({preprocessed_input.shape[1]})")
        # Create generic feature names if needed
        feature_names = [f'feature_{i}' for i in range(preprocessed_input.shape[1])]
        print(f"Created {len(feature_names)} generic feature names")

    # Convert to DataFrame for easier handling
    preprocessed_df = pd.DataFrame(preprocessed_input, columns=feature_names)
    print(f"Created DataFrame with shape: {preprocessed_df.shape}")

    # Define a prediction function for SHAP
    def predict_proba_fn(X):
        # Ensure X is a DataFrame with correct columns
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=input_data.columns)
        return model.predict_proba(X)

    # Use KernelExplainer for SHAP
    explainer = shap.KernelExplainer(predict_proba_fn, background_data, link="logit")
    shap_values = explainer.shap_values(input_data, nsamples=100)

    # For binary classification, use class 1 SHAP values
    if isinstance(shap_values, list):
        values = shap_values[1]  # For binary classification, use class 1
    else:
        values = shap_values

    # Ensure values is 1D array
    if len(values.shape) > 1:
        # If 2D, take mean across samples
        mean_abs_shap = np.abs(values).mean(axis=0)
        original_shap_values = values[0]  # Get first sample's values
    else:
        mean_abs_shap = np.abs(values)
        original_shap_values = values

    # Ensure all arrays are 1D
    feature_names = np.array(feature_names).flatten()
    mean_abs_shap = np.array(mean_abs_shap).flatten()
    original_shap_values = np.array(original_shap_values).flatten()

    # Debug prints to check array lengths
    print("Length feature_names:", len(feature_names))
    print("Length mean_abs_shap:", len(mean_abs_shap))
    print("Length original_shap_values:", len(original_shap_values))

    # 🔧 Array Length Correction
    # Trim all arrays to the shortest length to prevent pandas ValueError
    min_len = min(len(feature_names), len(mean_abs_shap), len(original_shap_values))
    if len(set([len(feature_names), len(mean_abs_shap), len(original_shap_values)])) != 1:
        print("Warning: array lengths mismatch → trimming to", min_len)
    feature_names = feature_names[:min_len]
    mean_abs_shap = mean_abs_shap[:min_len]
    original_shap_values = original_shap_values[:min_len]

    # Create DataFrame with feature names and SHAP values
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap,
        'Original_SHAP': original_shap_values
    })

    # Sort by absolute importance and get top 5
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(5)

    # Add effect direction using the stored original SHAP values
    feature_importance_df['Effect'] = feature_importance_df['Original_SHAP'].apply(lambda x: 'Increased' if x > 0 else 'Decreased')
    feature_importance_df['Direction'] = feature_importance_df['Original_SHAP'].apply(lambda x: '↑' if x > 0 else '↓')

    # Get the top feature and its importance
    top_feature = feature_importance_df.iloc[0]
    feature_name = top_feature['Feature']
    original_shap = top_feature['Original_SHAP']

    # Drop the Original_SHAP column as it was only used for effect direction
    feature_importance_df = feature_importance_df.drop('Original_SHAP', axis=1)

    # Clean up feature names for display
    feature_importance_df['Feature'] = feature_importance_df['Feature'].apply(lambda x: 
        x.replace('cat__', '')
        .replace('_', ' ')
        .replace('provider specialty', 'Provider Specialty:')
        .replace('payer', 'Payer:')
        .replace('diagnosis code', 'Diagnosis Code:')
        .replace('procedure code', 'Procedure Code:')
        .replace('patient gender', 'Patient Gender:')
        .replace('urgency flag', 'Urgency:')
        .replace('documentation complete', 'Documentation:')
        .replace('region', 'Region:')
    )

    # Clean up the feature name for the explanation narrative
    feature_name = (feature_name
        .replace('cat__', '')
        .replace('_', ' ')
        .replace('provider specialty', 'Provider Specialty:')
        .replace('payer', 'Payer:')
        .replace('diagnosis code', 'Diagnosis Code:')
        .replace('procedure code', 'Procedure Code:')
        .replace('patient gender', 'Patient Gender:')
        .replace('urgency flag', 'Urgency:')
        .replace('documentation complete', 'Documentation:')
        .replace('region', 'Region:')
    )

    # Generate user-friendly narrative explanation
    if feature_name.startswith('Provider Specialty'):
        specialty = feature_name.split(': ')[1]
        if original_shap > 0:
            explanation = f"Having a provider in the {specialty} specialty is associated with a higher chance of approval for this type of request, based on historical data."
        else:
            explanation = f"Having a provider in the {specialty} specialty is associated with a lower chance of approval for this type of request, based on historical data."
    elif feature_name.startswith('Payer'):
        payer = feature_name.split(': ')[1]
        if original_shap > 0:
            explanation = f"This insurance company ({payer}) has historically approved similar requests more often."
        else:
            explanation = f"This insurance company ({payer}) has historically approved similar requests less often."
    elif feature_name.startswith('Diagnosis Code'):
        code = feature_name.split(': ')[1]
        if original_shap > 0:
            explanation = f"The diagnosis code {code} is associated with higher approval rates for this type of request."
        else:
            explanation = f"The diagnosis code {code} is associated with lower approval rates for this type of request."
    else:
        if original_shap > 0:
            explanation = f"The factor {feature_name} is associated with higher approval rates for this type of request."
        else:
            explanation = f"The factor {feature_name} is associated with lower approval rates for this type of request."

    # Create full feature importance DataFrame
    full_feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap,
        'Original_SHAP': original_shap_values
    })
    
    # Clean up feature names for display
    full_feature_importance_df['Feature'] = full_feature_importance_df['Feature'].apply(lambda x: 
        x.replace('cat__', '')
        .replace('_', ' ')
        .replace('provider specialty', 'Provider Specialty:')
        .replace('payer', 'Payer:')
        .replace('diagnosis code', 'Diagnosis Code:')
        .replace('procedure code', 'Procedure Code:')
        .replace('patient gender', 'Patient Gender:')
        .replace('urgency flag', 'Urgency:')
        .replace('documentation complete', 'Documentation:')
        .replace('region', 'Region:')
    )
    
    # Add effect direction using the stored original SHAP values
    full_feature_importance_df['Effect'] = full_feature_importance_df['Original_SHAP'].apply(lambda x: 'Increased' if x > 0 else 'Decreased')
    full_feature_importance_df['Direction'] = full_feature_importance_df['Original_SHAP'].apply(lambda x: '↑' if x > 0 else '↓')
    
    # Drop the Original_SHAP column
    full_feature_importance_df = full_feature_importance_df.drop('Original_SHAP', axis=1)
    
    # Filter features based on importance
    full_feature_importance_df = full_feature_importance_df[
        (full_feature_importance_df['Importance'] > 0) |  # Feature had meaningful impact
        (abs(full_feature_importance_df['Importance']) >= 0.01)  # Feature had significant impact
    ]
    
    # Round Importance to 3 decimal places
    full_feature_importance_df['Importance'] = full_feature_importance_df['Importance'].round(3)
    
    # Sort by absolute importance
    full_feature_importance_df = full_feature_importance_df.sort_values('Importance', ascending=False)

    # Group features by category and sum their SHAP values
    feature_categories = {
        'Provider Specialty': 'provider specialty',
        'Diagnosis Code': 'diagnosis code',
        'Procedure Code': 'procedure code',
        'Payer': 'payer',
        'Patient Gender': 'patient gender',
        'Urgency': 'urgency flag',
        'Documentation': 'documentation complete',
        'Region': 'region'
    }
    
    grouped_importance = defaultdict(float)
    for feature, importance in zip(feature_names, mean_abs_shap):
        for category, prefix in feature_categories.items():
            if prefix in feature.lower():
                grouped_importance[category] += abs(importance)
                break
    
    # Create grouped importance DataFrame
    grouped_importance_df = pd.DataFrame({
        'Category': list(grouped_importance.keys()),
        'Importance': list(grouped_importance.values())
    }).sort_values('Importance', ascending=False)

    return feature_importance_df, explanation, original_shap, full_feature_importance_df, grouped_importance_df

# Initialize S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'pa-predictor-bucket-rs'
MODEL_FILE = 'pa_predictor_model.pkl'


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
            'patient_age': 50,
            'patient_gender': 'M',
            'procedure_code': '27447',
            'diagnosis_code': 'M17.11',
            'provider_specialty': 'Ortho Surgeon',
            'payer': 'Medicare',
            'urgency_flag': 'Y',
            'documentation_complete': 'Y',
            'prior_denials': 0,
            'region': 'Midwest'
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
                value=st.session_state.form_data['patient_age']
            )
            st.session_state.form_data['patient_gender'] = st.selectbox(
                "Patient Gender", 
                ['M', 'F'],
                index=['M', 'F'].index(st.session_state.form_data['patient_gender'])
            )
            # Procedure Code input and validation
            st.session_state.form_data['procedure_code'] = st.text_input(
                "Procedure Code (enter CPT code):",
                value=st.session_state.form_data['procedure_code']
            )
            input_proc_code = st.session_state.form_data['procedure_code'].strip().upper()
            if input_proc_code:
                if input_proc_code not in valid_cpt_codes:
                    st.warning(f"⚠️ The procedure code '{input_proc_code}' is not a valid CPT code. Please correct it before proceeding.")
                else:
                    st.success(f"✅ Procedure code '{input_proc_code}' is valid.")
        with col2:
            # Diagnosis Code input and validation (moved from Step 2)
            st.session_state.form_data['diagnosis_code'] = st.text_input(
                "Diagnosis Code",
                value=st.session_state.form_data['diagnosis_code'],
                max_chars=10,
                help="Enter any valid ICD-10 code (e.g., M17.11, K21.9, etc.)"
            )
            st.session_state.form_data['provider_specialty'] = st.selectbox(
                "Provider Specialty",
                valid_specialties,
                index=valid_specialties.index(st.session_state.form_data['provider_specialty'])
                if st.session_state.form_data['provider_specialty'] in valid_specialties else 0
)

            
            input_diag_code = st.session_state.form_data['diagnosis_code'].strip().upper()
            if input_diag_code:
                if input_diag_code not in valid_icd10_codes:
                    st.warning(f"⚠️ The diagnosis code '{input_diag_code}' is not a valid ICD-10 code. Please correct it before proceeding.")
                else:
                    st.success(f"✅ Diagnosis code '{input_diag_code}' is valid.")
        # Only enable Next Step if both codes are valid
        codes_valid = (
            input_proc_code in valid_cpt_codes and
            input_diag_code in valid_icd10_codes
        )
        if st.button("Next Step", key="next_step_1"):
            if codes_valid:
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Both procedure code and diagnosis code must be valid to proceed.")
    
    # Step 2: Request Details
    elif st.session_state.step == 2:
        st.markdown("""
            <div class="card-container">
                <div class="card-title">Request Details</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Load payers dynamically from dataset
            payer_df = pd.read_csv('synthetic_pa_dataset_v2.csv')
            payer_options = sorted(payer_df['payer'].dropna().unique().tolist())
            st.session_state.form_data['payer'] = st.selectbox(
                "Insurance Payer",
                payer_options,
                index=payer_options.index(st.session_state.form_data['payer']) if st.session_state.form_data['payer'] in payer_options else 0
            )
            st.session_state.form_data['urgency_flag'] = st.selectbox(
                "Urgent Request", 
                ['Y', 'N'],
                index=['Y', 'N'].index(st.session_state.form_data['urgency_flag'])
            )
        with col2:
            st.session_state.form_data['documentation_complete'] = st.selectbox(
                "Documentation Complete", 
                ['Y', 'N'],
                index=['Y', 'N'].index(st.session_state.form_data['documentation_complete'])
            )
            st.session_state.form_data['prior_denials'] = st.number_input(
                "Prior Denials", 
                min_value=0, 
                max_value=10, 
                value=st.session_state.form_data['prior_denials']
            )
            st.session_state.form_data['region'] = st.selectbox(
                "Region", 
                ['Midwest', 'Northeast', 'South', 'West'],
                index=['Midwest', 'Northeast', 'South', 'West'].index(st.session_state.form_data['region'])
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
                    model = load_model_from_s3()
                    if model is None:
                        st.error("Failed to load the model. Please try again later.")
                        st.stop()

                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0][1] * 100

                    # Load background dataset from synthetic_pa_dataset.csv, dropping the outcome column
                    background_data = pd.read_csv('synthetic_pa_dataset.csv').drop(columns=['outcome']).sample(100, random_state=42)

                    # Extract feature importance
                    feature_importance_df, explanation, original_shap, full_feature_importance_df, grouped_importance_df = get_feature_importance(model, input_data, background_data)

                    # Display in Streamlit
                    st.subheader("Top Influencing Encoded Features")
                    st.dataframe(feature_importance_df)

                    # Dynamic recommendation logic
                    if feature_importance_df.iloc[0]['Feature'].startswith('Provider Specialty'):
                        if prediction == 0:
                            dynamic_message = "⚠️ The provider specialty is driving the denial. Consider reviewing for specificity or documentation."
                        else:
                            dynamic_message = "✅ The provider specialty was a key approval factor."
                    elif feature_importance_df.iloc[0]['Feature'].startswith('Payer'):
                        if prediction == 0:
                            dynamic_message = "⚠️ The insurance payer is driving the denial. Consider peer review or alternative coding."
                        else:
                            dynamic_message = "✅ The insurance payer was a key approval factor."
                    elif feature_importance_df.iloc[0]['Feature'].startswith('Diagnosis Code'):
                        if prediction == 0:
                            dynamic_message = "⚠️ The diagnosis code is driving the denial. Consider reviewing for specificity or documentation."
                        else:
                            dynamic_message = "✅ The diagnosis code was a key approval factor."
                    else:
                        if prediction == 0:
                            dynamic_message = "ℹ️ Other factors are influencing the decision."
                        else:
                            dynamic_message = "✅ Other factors contributed positively to the approval."

                    st.info(dynamic_message)

                    # Add vertical spacing above the results section
                    st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)

                    # Prediction Results title in its own card
                    st.markdown("""
                        <div style="text-align:center; font-size:1.25rem; font-weight:600; margin-bottom:1rem; color:#0f172a;">
                        Prediction Results
                        </div>
                    """, unsafe_allow_html=True)

                    # Add vertical spacing between title and charts
                    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

                    # Side-by-side charts
                    chart_col1, chart_col2 = st.columns(2, gap="large")
                    
                    with chart_col1:
                        # Gauge Chart with subheader
                        st.subheader("Predicted Approval Probability")
                        fig = create_gauge_chart(probability)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        # Display narrative explanation
                        st.info(explanation)
                        
                        # Add SHAP value clarification with tooltip
                        st.caption("Understanding Feature Impact")
                        st.caption("ℹ️ Hover for details", help="SHAP values measure how much each factor pushed the approval probability up or down in this case.")
                        
                        # Add note about feature types
                        st.caption("Note: The model finds provider specialty and diagnosis code to be the most predictive features for prior authorization decisions, based on training data. Other factors may have smaller influence.")
                        
                        # Title for feature importance visualization
                        st.markdown("<div style='font-weight:600; font-size:1.15rem; margin-bottom:0.5rem;'>Top Influencing Factors</div>", unsafe_allow_html=True)
                        
                        # Create horizontal bar chart for top 5 features
                        fig = px.bar(
                            feature_importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Effect',  # Color by effect direction
                            color_discrete_map={'Increased': '#68d391', 'Decreased': '#fc8181'},
                            labels={'Importance': 'Impact on Approval', 'Feature': 'Factor'},
                            height=400,  # Increased height
                            text='Direction'  # Add direction arrows
                        )
                        fig.update_layout(
                            showlegend=True,
                            legend_title='Effect on Approval',
                            legend_orientation='v',
                            legend_y=1,
                            legend_x=1.02,
                            margin=dict(l=0, r=40, t=0, b=0),
                            xaxis_title="Impact on Approval",
                            yaxis_title="",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(size=12),  # Reduce overall font size
                        )
                        fig.update_xaxes(tickfont=dict(size=11), title_font=dict(size=12))
                        fig.update_yaxes(tickfont=dict(size=11), title_font=dict(size=12), automargin=True)
                        fig.update_traces(textposition='outside', cliponaxis=False)
                        st.plotly_chart(fig, use_container_width=True)

                    # Vertical spacing below charts
                    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

                    # Approval status badge in its own centered card with improved styling
                    status = "Approved" if prediction == 1 else "Denied"
                    status_class = "approved" if prediction == 1 else "denied"
                    status_emoji = "✅" if prediction == 1 else "❌"
                    st.markdown(f"""
                        <div class="card-container" style="max-width: 300px; margin: 0 auto 2rem auto; text-align: center;">
                            <div class="status-badge {status_class}" style="font-size:1.5rem; padding:1rem 2rem; display:inline-block; border-radius:8px;">
                                {status_emoji} <span style="margin-left: 0.5rem;">{status}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Recommendation info below status card with dynamic styling
                    recommendation = get_recommendation(st.session_state.form_data['documentation_complete'], st.session_state.form_data['prior_denials'])
                    if "Submit missing documentation" in recommendation:
                        st.warning(f"💡 Recommendation: {recommendation}")
                    elif "No additional action recommended" in recommendation:
                        st.success(f"💡 Recommendation: {recommendation}")
                    else:
                        st.error(f"💡 Recommendation: {recommendation}")

                    # Add conditional message based on top feature importance with tooltip
                    top_importance = feature_importance_df.iloc[0]['Importance']
                    if top_importance < 0.1:
                        st.info("No single factor strongly influenced this decision; approval is based on multiple small factors.")
                    else:
                        col1, col2 = st.columns([20, 1])
                        with col1:
                            st.info("Other factors also contributed to this decision.")
                        with col2:
                            st.caption("ℹ️", help="Other factors had lower impact but still influenced the prediction.")

                    # Pre-define display_df for the download button, so it always exists even if advanced view is not checked
                    display_df = full_feature_importance_df.copy()

                    # The following checkboxes do NOT reset st.session_state.step, so toggling them won't reset the app to the first page.
                    # This preserves the current step and keeps the user on the results page.
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        show_advanced = st.checkbox("Show full feature importances", key="show_advanced")
                    with col2:
                        show_all_features = st.checkbox("Show all input features (including zero impact)", key="show_all_features")

                    if show_advanced:
                        st.markdown("### Full Feature Importances")
                        st.caption("This table shows all features that contributed to this case.")
                        
                        # Filter features based on show_all_features setting
                        if not show_all_features:
                            display_df = display_df[display_df['Importance'] > 0.001]  # Only show meaningful impacts
                        
                        # Add effect direction to full feature importance DataFrame
                        display_df['Effect'] = ['Increased' if v > 0 else 'Decreased' for v in values[0][display_df.index]]
                        display_df['Direction'] = ['↑' if v > 0 else '↓' for v in values[0][display_df.index]]
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Show grouped feature importances
                        st.markdown("### Feature Impact by Category")
                        st.caption("This shows the total impact of each feature category on the prediction.")
                        fig_grouped = px.bar(
                            grouped_importance_df,
                            x='Importance',
                            y='Category',
                            orientation='h',
                            color='Category',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            labels={'Importance': 'Total Impact', 'Category': 'Feature Category'},
                            height=300
                        )
                        fig_grouped.update_layout(
                            showlegend=False,
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis_title="Total Impact",
                            yaxis_title="",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_grouped, use_container_width=True)

                    # Download button for prediction results
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=csv,
                        file_name="prediction_report.csv",
                        mime="text/csv",
                        help="Download detailed prediction results as CSV"
                    )

    # Footer
    st.markdown("""
        <div class="footer">
            © 2025 PA Predictor. For demonstration purposes only.<br>
            Contact: ravikirans723@gmail.com
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