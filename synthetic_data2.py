import pandas as pd
import numpy as np

def generate_synthetic_data(num_rows=5000):
    """
    Generate a synthetic prior authorization dataset for PA Predictor retraining.
    Returns a pandas DataFrame and saves it as 'synthetic_pa_dataset_v2.csv'.
    """
    np.random.seed(42)  # For reproducibility

    # Define possible values for each categorical feature
    genders = ['M', 'F']
    procedure_codes = ['27447', '29881', '99213', '12001', '36415']
    diagnosis_codes = ['M17.11', 'K21.9', 'I10', 'E11.9', 'N18.9']
    specialties = ['Ortho Surgeon', 'Cardiologist', 'Primary Care', 'Dermatologist', 'ENT']
    payers = ['Medicare', 'Aetna', 'Cigna', 'UnitedHealthcare']
    urgency_flags = ['Y', 'N']
    doc_complete_flags = ['Y', 'N']
    regions = ['Midwest', 'Northeast', 'South', 'West']

    data = []
    for _ in range(num_rows):
        patient_age = np.random.randint(18, 91)
        patient_gender = np.random.choice(genders)
        procedure_code = np.random.choice(procedure_codes)
        diagnosis_code = np.random.choice(diagnosis_codes)
        provider_specialty = np.random.choice(specialties)
        payer = np.random.choice(payers)
        urgency_flag = np.random.choice(urgency_flags)
        documentation_complete = np.random.choice(doc_complete_flags, p=[0.85, 0.15])  # More likely to be 'Y'
        prior_denials_provider = np.random.randint(0, 11)
        region = np.random.choice(regions)

        # Business logic for approval probability
        approval_prob = 0.8
        if documentation_complete == 'N':
            approval_prob -= 0.4
        if prior_denials_provider > 5:
            approval_prob -= 0.2
        if urgency_flag == 'Y':
            approval_prob -= 0.1
        if diagnosis_code == 'M17.11':
            approval_prob -= 0.1
        if procedure_code == '27447':
            approval_prob -= 0.05
        approval_prob = np.clip(approval_prob, 0, 1)

        # Simulate outcome
        outcome = np.random.binomial(1, approval_prob)

        data.append({
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'procedure_code': procedure_code,
            'diagnosis_code': diagnosis_code,
            'provider_specialty': provider_specialty,
            'payer': payer,
            'urgency_flag': urgency_flag,
            'documentation_complete': documentation_complete,
            'prior_denials_provider': prior_denials_provider,
            'region': region,
            'outcome': outcome
        })

    df = pd.DataFrame(data)
    df.to_csv('synthetic_pa_dataset_v2.csv', index=False)
    return df

# Example usage:
# df = generate_synthetic_data(5000)
# print(df.head())