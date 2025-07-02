# React Frontend Quick Setup

## 1. Create React Project

```bash
# Create new React project with TypeScript
npx create-react-app pa-predictor-frontend --template typescript
cd pa-predictor-frontend

# Install essential dependencies
npm install @headlessui/react @heroicons/react
npm install tailwindcss postcss autoprefixer
npm install axios react-hook-form @hookform/resolvers zod
npm install recharts framer-motion
npm install react-router-dom
npm install @types/node

# Initialize Tailwind CSS
npx tailwindcss init -p
```

## 2. Tailwind Configuration

```javascript
// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        healthcare: {
          blue: '#0ea5e9',
          green: '#10b981',
          red: '#ef4444',
        }
      },
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
```

## 3. Essential TypeScript Types

```typescript
// src/types/prediction.ts
export interface PredictionInput {
  patient_age: number;
  patient_gender: 'M' | 'F';
  procedure_code: string;
  diagnosis_code: string;
  provider_specialty: string;
  payer: string;
  urgency_flag: 'Y' | 'N';
  documentation_complete: 'Y' | 'N';
  prior_denials_provider: number;
  region: 'Midwest' | 'Northeast' | 'South' | 'West';
}

export interface PredictionResponse {
  approval_prediction: number;
  probability: number;
  confidence_score: number;
  model_version: string;
  prediction_id: string;
  timestamp: string;
  recommendations: string[];
}

export interface ValidationCodes {
  cpt_codes_count: number;
  icd10_codes_count: number;
  specialties_count: number;
  sample_cpt_codes: string[];
  sample_icd10_codes: string[];
  sample_specialties: string[];
}
```

## 4. API Service

```typescript
// src/services/api.ts
import axios from 'axios';
import { PredictionInput, PredictionResponse, ValidationCodes } from '../types/prediction';

const API_BASE_URL = 'http://54.163.203.207:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// Add request interceptor for analytics
api.interceptors.request.use((config) => {
  const sessionId = localStorage.getItem('sessionId') || generateSessionId();
  localStorage.setItem('sessionId', sessionId);
  
  config.headers['X-Session-ID'] = sessionId;
  config.headers['X-User-ID'] = 'anonymous'; // Replace with actual user ID when auth is added
  
  return config;
});

export const predictionAPI = {
  predict: async (data: PredictionInput): Promise<PredictionResponse> => {
    const response = await api.post('/predict', data);
    return response.data;
  },
  
  getValidationCodes: async (): Promise<ValidationCodes> => {
    const response = await api.get('/validation/codes');
    return response.data;
  },
  
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  }
};

function generateSessionId(): string {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}
```

## 5. Main App Component

```typescript
// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Navbar from './components/layout/Navbar';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Dashboard from './pages/Dashboard';
import About from './pages/About';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <main>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
```

## 6. Prediction Form Component

```typescript
// src/components/forms/PredictionForm.tsx
import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { PredictionInput } from '../../types/prediction';
import { predictionAPI } from '../../services/api';

const predictionSchema = z.object({
  patient_age: z.number().min(18).max(90),
  patient_gender: z.enum(['M', 'F']),
  procedure_code: z.string().min(1),
  diagnosis_code: z.string().min(1),
  provider_specialty: z.string().min(1),
  payer: z.string().min(1),
  urgency_flag: z.enum(['Y', 'N']),
  documentation_complete: z.enum(['Y', 'N']),
  prior_denials_provider: z.number().min(0).max(10),
  region: z.enum(['Midwest', 'Northeast', 'South', 'West']),
});

export default function PredictionForm() {
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<PredictionInput>({
    resolver: zodResolver(predictionSchema),
  });

  const onSubmit = async (data: PredictionInput) => {
    setIsLoading(true);
    try {
      const response = await predictionAPI.predict(data);
      setResult(response);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Prior Authorization Predictor
        </h2>
        
        {/* Step Indicator */}
        <div className="flex items-center justify-center mb-8">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            step >= 1 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
          }`}>
            1
          </div>
          <div className={`w-16 h-1 mx-2 ${
            step >= 2 ? 'bg-blue-600' : 'bg-gray-200'
          }`}></div>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            step >= 2 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
          }`}>
            2
          </div>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {step === 1 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Patient Information */}
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Patient Age
                </label>
                <input
                  type="number"
                  {...register('patient_age', { valueAsNumber: true })}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
                {errors.patient_age && (
                  <p className="mt-1 text-sm text-red-600">{errors.patient_age.message}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Patient Gender
                </label>
                <select
                  {...register('patient_gender')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value="">Select Gender</option>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
                {errors.patient_gender && (
                  <p className="mt-1 text-sm text-red-600">{errors.patient_gender.message}</p>
                )}
              </div>

              {/* Add more fields for step 1 */}
            </div>
          )}

          {step === 2 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Request Details */}
              {/* Add fields for step 2 */}
            </div>
          )}

          <div className="flex justify-between pt-6">
            {step > 1 && (
              <button
                type="button"
                onClick={() => setStep(step - 1)}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
              >
                Previous
              </button>
            )}
            
            {step < 2 ? (
              <button
                type="button"
                onClick={() => setStep(step + 1)}
                className="ml-auto px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Next
              </button>
            ) : (
              <button
                type="submit"
                disabled={isLoading}
                className="ml-auto px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
              >
                {isLoading ? 'Predicting...' : 'Get Prediction'}
              </button>
            )}
          </div>
        </form>

        {/* Results Display */}
        {result && (
          <div className="mt-8 p-6 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Approval Status</p>
                <p className={`text-lg font-bold ${
                  result.approval_prediction ? 'text-green-600' : 'text-red-600'
                }`}>
                  {result.approval_prediction ? 'Approved' : 'Denied'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Probability</p>
                <p className="text-lg font-bold text-blue-600">
                  {(result.probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            
            {result.recommendations && (
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-2">Recommendations:</p>
                <ul className="list-disc list-inside space-y-1">
                  {result.recommendations.map((rec: string, index: number) => (
                    <li key={index} className="text-sm text-gray-700">{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

## 7. Start Development

```bash
# Start the development server
npm start

# Build for production
npm run build
```

## 8. Next Steps

1. **Complete the form fields** for both steps
2. **Add validation** for CPT and ICD-10 codes
3. **Implement charts** for results visualization
4. **Add analytics tracking** for user interactions
5. **Style the components** with Tailwind CSS
6. **Add animations** with Framer Motion
7. **Deploy to Vercel/Netlify**

This gives you a solid foundation to start building your professional React frontend! 