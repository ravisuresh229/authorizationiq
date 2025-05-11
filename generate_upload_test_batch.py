import pandas as pd
import numpy as np
import boto3
from datetime import datetime

# Define your bucket and path
BUCKET_NAME = "pa-predictor-bucket-rs"
BATCH_FOLDER = "inputs/"

def generate_test_batch(n=100):
    np.random.seed(42)
    
    genders = ['M', 'F']
    procedure_codes = ['27447', '29881', '99213', '12001', '36415']
    diagnosis_codes = ['M17.11', 'K21.9', 'I10', 'E11.9', 'N18.9']
    specialties = ['Ortho Surgeon', 'Cardiologist', 'Primary Care', 'Dermatologist', 'ENT']
    payers = ['Medicare', 'Aetna', 'Cigna', 'UnitedHealthcare']
    urgencies = ['Y', 'N']
    docs = ['Y', 'N']
    regions = ['Midwest', 'Northeast', 'South', 'West']

    data = []
    for _ in range(n):
        row = {
            "patient_age": np.random.randint(18, 91),
            "patient_gender": np.random.choice(genders),
            "procedure_code": np.random.choice(procedure_codes),
            "diagnosis_code": np.random.choice(diagnosis_codes),
            "provider_specialty": np.random.choice(specialties),
            "payer": np.random.choice(payers),
            "urgency_flag": np.random.choice(urgencies),
            "documentation_complete": np.random.choice(docs, p=[0.85, 0.15]),
            "prior_denials_provider": np.random.randint(0, 11),
            "region": np.random.choice(regions)
        }
        data.append(row)

    return pd.DataFrame(data)

def upload_to_s3(df, bucket, folder):
    s3 = boto3.client("s3")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"{folder}batch_{timestamp}.csv"

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    print(f"✅ Uploaded batch to s3://{bucket}/{key}")

if __name__ == "__main__":
    df = generate_test_batch(100)
    upload_to_s3(df, BUCKET_NAME, BATCH_FOLDER) 