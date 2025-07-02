import { PredictionInput, PredictionResult, RecentPrediction } from '../types/prediction';

// Use Render.com URL for production, localhost for development
const API_BASE_URL = process.env.REACT_APP_API_URL || 
  (window.location.hostname === 'localhost' 
    ? 'http://localhost:8001' 
    : 'https://pa-predictor-api.onrender.com'); // Replace with your actual Render URL

export const api = {
  async predict(input: PredictionInput): Promise<PredictionResult> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Prediction failed: ${response.statusText}`);
    }

    return response.json();
  },

  async getRecentPredictions(): Promise<RecentPrediction[]> {
    const response = await fetch(`${API_BASE_URL}/recent-predictions`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch recent predictions: ${response.statusText}`);
    }

    return response.json();
  },

  async getHealth(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  },

  async getAbout(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/about`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch about info: ${response.statusText}`);
    }

    return response.json();
  },

  async getPayers(): Promise<string[]> {
    const response = await fetch(`${API_BASE_URL}/codes/payers`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch payers: ${response.statusText}`);
    }

    return response.json();
  }
}; 