import pandas as pd
import glob
import pickle
import boto3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT_DIR = "inputs"
S3_BUCKET = "pa-predictor-bucket-rs"
MODEL_FILENAME = "latest_pipeline.pkl"
ARCHIVE_PREFIX = "archive/"

s3 = boto3.client("s3")

def load_all_batches():
    files = glob.glob(f"{INPUT_DIR}/batch_*.csv")
    dfs = [pd.read_csv(file) for file in files]
    return pd.concat(dfs, ignore_index=True), files

def prepare_data(df):
    df['label'] = ((df['documentation_complete'] == 'Y') & (df['prior_denials_provider'] <= 5)).astype(int)
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label']
    return X, y

def build_pipeline(X):
    categorical = X.select_dtypes(include=['object']).columns.tolist()
    numerical = X.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', StandardScaler(), numerical)
    ])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    return Pipeline([('preprocessor', preprocessor), ('clf', clf)])

def save_to_s3(pipeline):
    pickle_bytes = pickle.dumps(pipeline)
    s3.put_object(Bucket=S3_BUCKET, Key=MODEL_FILENAME, Body=pickle_bytes)
    print(f"✅ Model retrained and uploaded to s3://{S3_BUCKET}/{MODEL_FILENAME}")

def archive_batches(s3_batch_keys):
    for key in s3_batch_keys:
        archive_key = ARCHIVE_PREFIX + os.path.basename(key)
        s3.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': key}, Key=archive_key)
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        print(f"📦 Archived {key} → {archive_key}")

def main():
    df, local_files = load_all_batches()
    X, y = prepare_data(df)
    pipeline = build_pipeline(X)
    pipeline.fit(X, y)
    save_to_s3(pipeline)

    # Archive S3 versions of all used batches
    s3_keys = [f"inputs/{os.path.basename(f)}" for f in local_files]
    archive_batches(s3_keys)

if __name__ == "__main__":
    main() 