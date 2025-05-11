#!/bin/bash
# Generates new batch and checks if retraining is needed

echo "🕒 Running hourly batch + retrain check @ $(date)"
python3 generate_upload_test_batch.py
python3 auto_retrain_if_ready.py 