import { CPTCode, ICD10Code } from '../types/prediction';

const API_BASE_URL = 'http://localhost:8001';

export const dataService = {
  async loadCPTCodes(): Promise<CPTCode[]> {
    const response = await fetch(`${API_BASE_URL}/codes/cpt`);
    if (!response.ok) throw new Error('Failed to fetch CPT codes');
    return response.json();
  },

  async loadICD10Codes(): Promise<ICD10Code[]> {
    const response = await fetch(`${API_BASE_URL}/codes/icd10`);
    if (!response.ok) throw new Error('Failed to fetch ICD-10 codes');
    return response.json();
  },

  async loadSpecialties(): Promise<string[]> {
    const response = await fetch(`${API_BASE_URL}/codes/specialties`);
    if (!response.ok) throw new Error('Failed to fetch specialties');
    return response.json();
  },

  async loadPayers(): Promise<string[]> {
    const response = await fetch(`${API_BASE_URL}/codes/payers`);
    if (!response.ok) throw new Error('Failed to fetch payers');
    return response.json();
  },

  async loadAbout() {
    const response = await fetch(`${API_BASE_URL}/about`);
    if (!response.ok) throw new Error('Failed to fetch about info');
    return response.json();
  }
}; 