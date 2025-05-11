import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import boto3
import pickle
import io
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'pa-predictor-bucket-rs'
MODEL_FILE = 'pa_predictor_model.pkl'

def load_data_from_s3():
    """
    Load the expanded synthetic dataset from S3 into a pandas DataFrame.
    This dataset now includes the full set of ICD-10 and CPT codes.
    """
    try:
        print("Loading data from S3...")
        response = s3.get_object(Bucket=BUCKET_NAME, Key='synthetic_pa_dataset_v2.csv')
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        print("Successfully loaded data from S3")
        
        # Print detailed dataset statistics
        print("\nDataset Statistics:")
        print(f"Total rows: {len(df)}")
        print(f"Unique diagnosis codes: {df['diagnosis_code'].nunique()}")
        print("\nTop 5 most frequent diagnosis codes:")
        print(df['diagnosis_code'].value_counts().head())
        print(f"\nUnique procedure codes: {df['procedure_code'].nunique()}")
        print("\nTop 5 most frequent procedure codes:")
        print(df['procedure_code'].value_counts().head())
        
        return df
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise

def preprocess_data(df):
    """Preprocess the data and prepare it for modeling"""
    # Separate features and target
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    
    # Define categorical columns
    categorical_cols = [
        'patient_gender', 'procedure_code', 'diagnosis_code',
        'provider_specialty', 'payer', 'urgency_flag',
        'documentation_complete', 'region'
    ]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', ['patient_age', 'prior_denials_provider'])
        ])
    
    return X, y, preprocessor

def train_model(X, y, preprocessor):
    """Train the model and return it along with test data"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, X_test, y_test

def save_model_to_s3(model):
    """Save the model to S3"""
    try:
        print("\nSaving model to S3...")
        # Save model to bytes
        model_bytes = pickle.dumps(model)
        
        # Upload to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=MODEL_FILE,
            Body=model_bytes
        )
        print(f"Model successfully saved to s3://{BUCKET_NAME}/{MODEL_FILE}")
    except Exception as e:
        print(f"Error saving model to S3: {e}")
        raise

def main():
    # Load data
    df = load_data_from_s3()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y, preprocessor = preprocess_data(df)
    
    # Train model and get metrics
    model, X_test, y_test = train_model(X, y, preprocessor)
    
    # Save model to S3
    save_model_to_s3(model)

if __name__ == "__main__":
    main() 