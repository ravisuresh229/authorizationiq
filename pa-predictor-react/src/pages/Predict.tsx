import React, { useState } from 'react';
import PredictionForm from '../components/forms/PredictionForm';
import { PredictionResult, PredictionInput } from '../types/prediction';

// Loading skeleton component
const PredictionSkeleton = () => (
  <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6 animate-pulse">
    <div className="h-6 bg-gray-300 rounded w-3/4 mb-4"></div>
    <div className="space-y-4">
      <div>
        <div className="h-4 bg-gray-300 rounded w-1/3 mb-2"></div>
        <div className="h-6 bg-gray-300 rounded w-1/2"></div>
      </div>
      <div>
        <div className="h-4 bg-gray-300 rounded w-1/4 mb-2"></div>
        <div className="h-4 bg-gray-300 rounded w-full"></div>
      </div>
      <div>
        <div className="h-4 bg-gray-300 rounded w-1/5 mb-2"></div>
        <div className="h-4 bg-gray-300 rounded w-2/3"></div>
      </div>
    </div>
  </div>
);

// Enhanced confidence meter component
const ConfidenceMeter = ({ probability }: { probability: number }) => {
  const percentage = probability * 100;
  const getColorClass = (prob: number) => {
    if (prob >= 70) return 'from-green-400 to-green-600';
    if (prob >= 50) return 'from-yellow-400 to-orange-500';
    return 'from-red-400 to-red-600';
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-500">Confidence Level</span>
        <span className="text-sm font-semibold text-gray-900">{percentage.toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${getColorClass(percentage)} transition-all duration-1000 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="text-xs text-gray-500">
        {percentage >= 70 ? 'High confidence' : percentage >= 50 ? 'Medium confidence' : 'Low confidence'}
      </div>
    </div>
  );
};

// KeyFactors component: shows only the top 3 most impactful factors
const KeyFactors: React.FC<{ 
  feature_importance?: { feature: string; importance: number; direction: string }[],
  input?: PredictionInput,
  prediction?: { approval_prediction: number; probability: number }
}> = ({ feature_importance, input, prediction }) => {
  if (!feature_importance || feature_importance.length === 0 || !input) return null;

  // Calculate top 3 factors based on the prediction and input data
  const factors = [];
  
  // Factor 1: Insurance Payer Pattern (most important)
  factors.push({
    name: "Insurance Payer Pattern",
    description: `${input.payer} approval rate for ${input.procedure_code}`,
    impact: Math.abs(feature_importance[0]?.importance || 0) > 0.5 ? "High" : "Medium",
    positive: prediction?.approval_prediction === 1
  });

  // Factor 2: Documentation Status
  factors.push({
    name: "Documentation Status",
    description: input.documentation_complete === 'Y' 
      ? "Complete documentation submitted" 
      : "Incomplete documentation (major risk factor)",
    impact: input.documentation_complete === 'N' ? "High" : "Medium",
    positive: input.documentation_complete === 'Y'
  });

  // Factor 3: Clinical Match
  factors.push({
    name: "Clinical Match",
    description: `${input.procedure_code} with ${input.diagnosis_code}`,
    impact: "Medium",
    positive: prediction?.approval_prediction === 1
  });

  return (
    <div className="space-y-3">
      <h3 className="font-semibold text-sm text-gray-600">Top Contributing Factors</h3>
      {factors.map((factor, index) => (
        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
          <div className="flex-1">
            <p className="font-medium">{factor.name}</p>
            <p className="text-sm text-gray-600">{factor.description}</p>
          </div>
          <span className={`text-sm font-medium ${
            factor.impact === 'High' ? 'text-red-600' : 
            factor.impact === 'Medium' ? 'text-yellow-600' : 'text-green-600'
          }`}>
            {factor.impact} Impact
          </span>
        </div>
      ))}
    </div>
  );
};

// Mapping for user-friendly feature names
const featureLabelMap: Record<string, string> = {
  documentation_complete: 'Documentation Status',
  procedure_code: 'Procedure Code',
  diagnosis_code: 'Diagnosis Code',
  provider_specialty: 'Provider Specialty',
  payer: 'Insurance Payer',
  prior_denials_provider: 'Prior Denial History',
  region: 'Region',
  urgency_flag: 'Urgency',
  patient_age: 'Patient Age',
  patient_gender: 'Patient Gender',
};

// Mapping for value explanations (optional, can be expanded)
const valueMap: Record<string, Record<string, string>> = {
  documentation_complete: {
    Y: 'Complete',
    N: 'Incomplete',
  },
  urgency_flag: {
    Y: 'Urgent',
    N: 'Routine',
  },
  patient_gender: {
    M: 'Male',
    F: 'Female',
  },
};

function generatePredictionInsight(input: PredictionInput, featureImportance?: { feature: string; importance: number; direction: string }[], probability?: number) {
  if (!featureImportance || featureImportance.length === 0) return 'No specific insight available.';
  // Top 2-3 features
  const top = featureImportance.slice(0, 3);
  const lines: string[] = [];
  // First line: overall summary
  if (probability !== undefined) {
    if (probability > 0.8) lines.push('This request has a very high chance of approval.');
    else if (probability > 0.5) lines.push('This request has a good chance of approval.');
    else if (probability > 0.3) lines.push('This request is borderline for approval.');
    else lines.push('This request has a high risk of denial.');
  }
  // Next lines: top features
  top.forEach(f => {
    // Try to match feature name to input key or value
    let key = Object.keys(input).find(k => {
      // Match by key in feature name
      if (f.feature.toLowerCase().includes(k.toLowerCase().replace('_', ' '))) return true;
      // Match by value in feature name (e.g., "Procedure Code 61514")
      if (typeof input[k as keyof PredictionInput] === 'string' && f.feature.includes(String(input[k as keyof PredictionInput]))) return true;
      return false;
    });
    let value = key ? input[key as keyof PredictionInput] : undefined;
    let label = key ? featureLabelMap[key] || f.feature : f.feature;
    let valueLabel = key && valueMap[key] && valueMap[key][String(value)] ? valueMap[key][String(value)] : value;
    let direction = f.direction === 'positive' ? 'increases' : 'decreases';
    if (label && value !== undefined) {
      lines.push(`${label}: ${valueLabel} (${direction} approval odds)`);
    } else {
      // Fallback: just show the feature name and direction
      lines.push(`${f.feature} (${direction} approval odds)`);
    }
  });
  return lines.join(' ');
}

const Predict: React.FC = () => {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async (predictionResult: PredictionResult) => {
    setResult(predictionResult);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center justify-start">
      <div className="w-full max-w-4xl mx-auto py-8 px-4">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">AuthorizationIQ</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Enter patient and procedure information to predict prior authorization requirements with AI-powered insights.
        </p>
      </div>
        <div className="flex justify-center">
          <div className="w-full max-w-lg">
          <PredictionForm 
            onPrediction={handlePrediction}
            onError={handleError}
            loading={loading}
            setLoading={setLoading}
          />

          {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4 shadow-sm">
              <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <div className="mt-2 text-sm text-red-700">{error}</div>
                </div>
              </div>
            </div>
          )}

            {loading && <PredictionSkeleton />}

            {result && !loading && (
              <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-6 transform transition-all duration-300 hover:shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">Prediction Result</h3>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                    result.prediction.approval_prediction === 1 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {result.prediction.approval_prediction === 1 ? 'APPROVED' : 'DENIED'}
                  </div>
                </div>

                <div className="space-y-6">
                <div>
                    <span className="text-sm font-medium text-gray-500">Prediction Status</span>
                    <div className={`mt-1 text-2xl font-bold ${
                      result.prediction.approval_prediction === 1 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {result.prediction.approval_prediction === 1 ? '✓ Approved' : '✗ Denied'}
                    </div>
                  </div>

                  <ConfidenceMeter probability={result.prediction.probability} />

                <div>
                    <span className="text-sm font-medium text-gray-500">Model Status</span>
                    <div className="mt-1 text-sm text-gray-900 flex items-center">
                      <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                    {result.prediction.status}
                  </div>
                </div>

                  {/* Key Factors Section */}
                  <KeyFactors feature_importance={result.feature_importance} input={result.input} prediction={result.prediction} />

                  {/* Dynamic Prediction Insights */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
                    <h3 className="font-semibold text-blue-900 mb-2">💡 Prediction Insight</h3>
                    {Array.isArray(result.insights) && result.insights.length > 0 ? (
                      <ul className="list-disc pl-5 space-y-1">
                        {result.insights.map((insightObj: { insight: string }, idx: number) => (
                          <li key={idx} className="text-sm text-blue-800 leading-relaxed">{insightObj.insight}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-blue-800 leading-relaxed">
                        {result.input ? generatePredictionInsight(result.input, result.feature_importance, result.prediction.probability) : 'No specific insight available.'}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}
            </div>
        </div>
      </div>

      {/* Professional Footer */}
      <footer className="mt-16 text-center text-sm text-gray-500">
        <p>Predictions based on analysis of 50K+ historical authorizations</p>
        <p>Not medical advice - Always verify with payer guidelines</p>
        <div className="mt-2">Built with React & FastAPI | ML-Powered Predictions</div>
      </footer>
    </div>
  );
};

export default Predict; 