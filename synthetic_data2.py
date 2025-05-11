import pandas as pd
import numpy as np

def generate_synthetic_data(num_rows=5000):
    """
    Generate a synthetic prior authorization dataset for PA Predictor retraining.
    Returns a pandas DataFrame and saves it as 'synthetic_pa_dataset_v2.csv'.
    """
    np.random.seed(42)  # For reproducibility

    # Load CPT codes from cpt_codes.csv
    cpt_df = pd.read_csv('cpt_codes.csv')
    procedure_codes = cpt_df['code'].tolist()
    print(f"Loaded {len(procedure_codes)} CPT codes for sampling")

    # Load ICD-10 codes from icd10_codes.csv
    icd_df = pd.read_csv('icd10_codes.csv')
    diagnosis_codes = icd_df['code'].tolist()
    print(f"Loaded {len(diagnosis_codes)} ICD-10 codes for synthetic data generation")

    # Define possible values for each categorical feature
    genders = ['M', 'F']
    specialties = ['Ortho Surgeon', 'Cardiologist', 'Primary Care', 'Dermatologist', 'ENT']
    payer_list = [
        'Aetna', 'Cigna', 'Medicare', 'UnitedHealthcare', 'Humana',
        'Blue Cross Blue Shield', 'Kaiser Permanente', 'Anthem',
        'Molina Healthcare', 'WellCare', 'Centene', 'Health Net',
        'Highmark', 'EmblemHealth', 'Oscar Health', 'CareSource',
        'Tufts Health Plan', 'Harvard Pilgrim', 'Priority Health',
        'Geisinger Health Plan'
    ]
    print(f"Using {len(payer_list)} unique insurance payers for synthetic data generation")
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
        payer = np.random.choice(payer_list)
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
        if diagnosis_code == 'M17.11':  # Keep this specific code check as it's a common diagnosis
            approval_prob -= 0.1
        if procedure_code == '27447':  # Keep this specific code check as it's a common procedure
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

if __name__ == '__main__':
    df = generate_synthetic_data(5000)
    print(f"\nGenerated synthetic dataset with {len(df)} rows")
    print("\nSample of generated data:")
    print(df.head())

def process_icd10_codes():
    """
    Process the ICD-10 codes from icd10_2025.txt and save them to icd10_codes.csv.
    Each line in the input file should be in the format: 'CODE Description'
    """
    # List to store the processed codes and descriptions
    codes_data = []
    
    # Read the input file line by line
    with open('icd10_2025.txt', 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the line into code and description
            # First word is the code, rest is the description
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                code = parts[0]
                description = parts[1]
                codes_data.append({'code': code, 'description': description})
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(codes_data)
    df.to_csv('icd10_codes.csv', index=False)
    return df

# @st.cache_data
def load_icd10_codes():
    """
    Load ICD-10 codes from the CSV file.
    Returns a set of valid codes for validation.
    """
    # First process the codes from the text file
    process_icd10_codes()
    
    # Then load the processed codes
    df = pd.read_csv('icd10_codes.csv')
    valid_codes = set(df['code'].str.strip().str.upper())
    return valid_codes