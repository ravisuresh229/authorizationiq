import pandas as pd
import random

# Possible values
ages = list(range(18, 90))
genders = ['M', 'F']
procedure_codes = ['27447', '43235', '99213', '11042', '47562']  # Example CPT codes
diagnosis_codes = ['M17.11', 'K21.9', 'I10', 'E11.9', 'N40.0']   # Example ICD-10 codes
specialties = ['Ortho Surgeon', 'GI Specialist', 'Primary Care', 'General Surgeon']
payers = ['Medicare', 'Aetna', 'UnitedHealthcare', 'Cigna']
urgencies = ['Y', 'N']
docs_complete = ['Y', 'N']
regions = ['Midwest', 'Northeast', 'South', 'West']

data = []

for _ in range(10000):
    age = random.choice(ages)
    gender = random.choice(genders)
    proc = random.choice(procedure_codes)
    diag = random.choice(diagnosis_codes)
    specialty = random.choice(specialties)
    payer = random.choice(payers)
    urgency = random.choice(urgencies)
    docs = random.choice(docs_complete)
    prior_denials = random.randint(0, 10)
    region = random.choice(regions)
    
    # Simple logic to bias outcome
    if docs == 'N':
        outcome = 0  # Denied
    elif prior_denials > 5:
        outcome = 0  # Denied
    elif urgency == 'Y' and docs == 'Y':
        outcome = 1  # Approved
    else:
        outcome = random.choices([0,1], weights=[0.3,0.7])[0]
    
    data.append([age, gender, proc, diag, specialty, payer, urgency, docs, prior_denials, region, outcome])

df = pd.DataFrame(data, columns=['patient_age','patient_gender','procedure_code','diagnosis_code','provider_specialty','payer','urgency_flag','documentation_complete','prior_denials_provider','region','outcome'])

df.to_csv('synthetic_pa_dataset.csv', index=False)

print("CSV file created: synthetic_pa_dataset.csv") 