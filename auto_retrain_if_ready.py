import os
import glob
import subprocess

BATCH_DIR = "inputs"
RETRAIN_SCRIPT = "retrain_model_from_batches.py"

def count_batches():
    return len(glob.glob(os.path.join(BATCH_DIR, "batch_*.csv")))

def main():
    batch_count = count_batches()
    print(f"🧮 Found {batch_count} batches...")
    if batch_count >= 5:
        print("🚀 Threshold met. Retraining model...")
        subprocess.run(["python3", RETRAIN_SCRIPT])
    else:
        print("⏳ Not enough batches yet. Skipping retraining.")

if __name__ == "__main__":
    main() 