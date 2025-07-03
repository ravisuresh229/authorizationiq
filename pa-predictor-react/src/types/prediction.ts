export interface PredictionInput {
  patient_age: number;
  patient_gender: "M" | "F";
  procedure_code: string;
  diagnosis_code: string;
  provider_specialty: string;
  payer: string;
  urgency_flag: "Y" | "N";
  documentation_complete: "Y" | "N";
  prior_denials_provider: number;
  region: "Midwest" | "Northeast" | "South" | "West";
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';
}

export interface PredictionResult {
  prediction: {
    approval_prediction: number;
    probability: number;
    status: string;
  };
  feature_importance?: FeatureImportance[];
  input?: PredictionInput;
  insights?: { insight: string; [key: string]: any }[];
}



export interface CPTCode {
  code: string;
  description: string;
}

export interface ICD10Code {
  code: string;
  description: string;
} 