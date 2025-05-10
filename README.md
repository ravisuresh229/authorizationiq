# 🏥 PA Approval Predictor

An AI-powered Prior Authorization (PA) approval prediction tool built using **Streamlit**, **SHAP explainability**, and **machine learning** to help healthcare providers estimate the likelihood of insurance approval for medical procedures.

## 🚀 Overview

Prior authorization processes cause delays and financial losses across healthcare systems. This tool uses predictive modeling to estimate approval likelihood based on CPT codes, ICD-10 codes, and other key factors.

Key features:

- ✅ Interactive web app built with **Streamlit**
- ✅ Visual explanations of predictions with **SHAP (SHapley values)**
- ✅ Real-time validation of CPT and ICD-10 codes
- ✅ Interactive charts and visualizations via **Plotly**
- ✅ Integration with **AWS S3** for data storage (optional)
- ✅ Modular code designed for expansion and integration into healthcare workflows

## 📊 Example Use Case

1. User inputs CPT code, ICD-10 diagnosis, and other procedure details
2. Model predicts the probability of approval
3. SHAP visualization explains the top contributing factors
4. Results can inform pre-submission reviews and documentation improvements

## 🏗️ Tech Stack

- Python
- Streamlit
- pandas, numpy
- scikit-learn
- SHAP
- Plotly
- boto3
- Streamlit-extras and other UI plugins

## 📷 Screenshots

_Add screenshots or GIFs of your app here to show the interface_

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pa-approval-predictor.git
cd pa-approval-predictor

2. Install Dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app.py

📁 Project Structure

├── app.py
├── requirements.txt
├── README.md
├── /data
├── /utils


