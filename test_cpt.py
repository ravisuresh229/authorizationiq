import pandas as pd

def process_cpt_codes():
    """
    Process the CPT codes from cpt4.csv and save them to cpt_codes.csv.
    Cleans the codes by stripping whitespace, converting to uppercase,
    removing duplicates and empty codes.
    """
    # Read the input file
    df = pd.read_csv('cpt4.csv')
    
    # Rename columns to match our expected format
    df = df.rename(columns={
        'com.medigy.persist.reference.type.clincial.CPT.code': 'code',
        'label': 'description'
    })
    
    # Clean the codes
    df['code'] = df['code'].astype(str).str.strip().str.upper()
    
    # Drop duplicates and empty codes
    df = df.dropna(subset=['code'])
    df = df.drop_duplicates(subset=['code'])
    
    # Save to CSV
    df.to_csv('cpt_codes.csv', index=False)
    return df

if __name__ == "__main__":
    # Process the codes
    df = process_cpt_codes()
    print(f'Processed {len(df)} CPT codes')
    print('\nFirst 5 rows:')
    print(df.head())
    
    # Test some validation
    test_codes = ['27447', '99213', 'INVALID']
    print('\nTesting validation:')
    for code in test_codes:
        is_valid = code in set(df['code'])
        print(f'Code {code}: {"✅ Valid" if is_valid else "❌ Invalid"}') 