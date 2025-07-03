import { CPTCode, ICD10Code } from '../types/prediction';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://pa-predictor-api-production-8341.up.railway.app';

export const dataService = {
  async loadCPTCodes(): Promise<CPTCode[]> {
    try {
      // Try API first, fallback to public files
      const response = await fetch(`${API_BASE_URL}/codes/cpt`);
      if (response.ok) return response.json();
    } catch (error) {
      console.log('API endpoint not available, using public files');
    }
    
    // Fallback to public files
    const response = await fetch('/cpt_codes.csv');
    if (!response.ok) throw new Error('Failed to fetch CPT codes');
    const csvText = await response.text();
    const lines = csvText.split('\n').slice(1); // Skip header
    return lines
      .filter(line => line.trim())
      .map(line => {
        const [code, description] = line.split(',').map(s => s.trim().replace(/"/g, ''));
        return { code, description };
      });
  },

  async loadICD10Codes(): Promise<ICD10Code[]> {
    try {
      // Try API first, fallback to public files
      const response = await fetch(`${API_BASE_URL}/codes/icd10`);
      if (response.ok) return response.json();
    } catch (error) {
      console.log('API endpoint not available, using public files');
    }
    
    // Fallback to public files
    const response = await fetch('/icd10_codes.csv');
    if (!response.ok) throw new Error('Failed to fetch ICD-10 codes');
    const csvText = await response.text();
    const lines = csvText.split('\n').slice(1); // Skip header
    return lines
      .filter(line => line.trim())
      .map(line => {
        const [code, description] = line.split(',').map(s => s.trim().replace(/"/g, ''));
        return { code, description };
      });
  },

  async loadSpecialties(): Promise<string[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/codes/specialties`);
      if (response.ok) return response.json();
    } catch (error) {
      console.log('API endpoint not available, using fallback specialties list');
    }
    // Fallback to hardcoded list if backend fails
    return [
      'Cardiology', 'Dermatology', 'Endocrinology', 'Gastroenterology', 
      'General Surgery', 'Internal Medicine', 'Neurology', 'Oncology', 
      'Orthopedics', 'Pediatrics', 'Psychiatry', 'Radiology', 'Urology'
    ];
  },

  async loadPayers(): Promise<string[]> {
    // Return common payers since we don't have this data
    return [
      'Aetna', 'Anthem', 'Blue Cross Blue Shield', 'Cigna', 
      'Humana', 'Kaiser Permanente', 'Medicare', 'UnitedHealthcare'
    ];
  },

  async loadAbout() {
    try {
      const response = await fetch(`${API_BASE_URL}/about`);
      if (response.ok) return response.json();
    } catch (error) {
      console.log('API endpoint not available, using fallback');
    }
    
    // Fallback about info
    return {
      title: "About PA Predictor",
      description: "The Prior Authorization (PA) Predictor is an AI-powered web application that predicts the likelihood of prior authorization approval for medical procedures.",
      features: [
        "Cloud-integrated code validation (CPT, ICD-10, specialties)",
        "ML model health monitoring",
        "Modern, user-friendly UI",
        "No free text for codes—only valid selections allowed",
        "End-to-end compatibility with backend ML model"
      ]
    };
  }
}; 