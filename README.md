# 🏥 PA Approval Predictor

An AI-powered web application that predicts the probability of **prior authorization (PA)** approval for medical procedures using real-world patient and request features.

Built with:
- 🧠 Machine Learning (scikit-learn)
- 🧾 SHAP interpretability
- 🚀 FastAPI backend (auto-reloads model from S3 every 60s)
- 🌐 Streamlit frontend (fully interactive, styled, multi-step form)
- ☁️ AWS S3 integration for live model hosting + prediction logging

---

## 🔍 Overview

Prior authorization delays cost the U.S. healthcare system **$50+ billion annually**. This tool aims to improve decision-making at the point-of-care by predicting the likelihood of PA approval based on key features like:

- Patient demographics (age, gender)
- CPT & ICD-10 codes
- Provider specialty
- Payer
- Documentation status
- Urgency & denial history

---

## 🧠 Machine Learning Model

- Trained on a **synthetic dataset** of 10,000+ cases
- Uses a **Random Forest classifier** within a scikit-learn pipeline
- Automatically encodes categorical features
- Uploaded to **AWS S3**, and reloaded by the FastAPI backend every 60s

---

## 🖥️ Live Demo

➡️ Streamlit Cloud App: **[https://...your-url...](https://...)**

---

## 🚀 Features

### ✅ Streamlit Frontend
- Multi-step form with input validation
- Dynamic CPT & ICD-10 code validation (74k+ and 8k+ codes)
- Animated UI, plotly gauge & bar charts
- SHAP feature importance + auto-generated recommendations

### 🧩 FastAPI Backend
- `/predict` endpoint exposed on EC2 instance
- Health check and input schema validation
- Model reloads from S3 every 60 seconds

### ☁️ AWS Integration
- **S3 buckets**:
  - `pa-predictor-bucket-rs` – model storage
  - `pa-predictor-logs` – prediction logging
- Model and logs handled with `boto3`

---

## 📦 Project Structure

```bash
.
├── app.py                    # Streamlit frontend
├── server.py                 # FastAPI backend
├── requirements.txt          # Dependencies
├── cpt_codes.csv             # Cleaned CPT codes
├── icd10_codes.csv           # Cleaned ICD-10 codes
├── provider_specialties.csv  # Valid specialties
├── .gitignore
### 🛠 Setup Instructions
Clone the repo

git clone https://github.com/ravisuresh229/pa-approval-predictor.git
cd pa-approval-predictor

Install dependencies
pip install -r requirements.txt

Run the app locally

streamlit run app.py
Run the backend (optional)

uvicorn server:app --reload --host 0.0.0.0 --port 8001

🔐 Environment Notes
Set your AWS credentials using ~/.aws/credentials or environment variables

Make sure the .venv/ and model pickle files are excluded from version control (.gitignore)

🙋‍♂️ Contact
Built with ❤️ by Ravi Suresh

For questions or feedback, feel free to reach out.

📈 Future Improvements
Real-time retraining pipeline from logged cases

HIPAA-compliant integration with EHR systems

Dynamic CPT/ICD-10 code suggestions via embeddings

