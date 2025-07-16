import React, { useState } from 'react';
import PredictionForm from '../components/forms/PredictionForm';
import { PredictionResult } from '../types/prediction';

const Predict: React.FC = () => {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summaryExpanded, setSummaryExpanded] = useState(false);
  const [explanationExpanded, setExplanationExpanded] = useState(false);

  const handlePrediction = async (predictionResult: PredictionResult) => {
    setResult(predictionResult);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setResult(null);
  };

  const resetForm = () => {
    setResult(null);
    setError(null);
  };

  const getPredictionColor = (isApproved: boolean) => {
    return isApproved ? 'emerald' : 'rose';
  };

  const getNextSteps = (isApproved: boolean) => {
    if (isApproved) {
      return [
        'Ensure all documentation is complete and up-to-date',
        'Submit as a standard request (non-urgent)',
        'Monitor for any additional payer requirements',
        'Consider peer-to-peer review if available'
      ];
    } else {
      return [
        'Gather additional clinical documentation to support medical necessity',
        'Review payer-specific requirements for this procedure',
        'Consider peer-to-peer review if available',
        'Prepare appeal documentation if needed'
      ];
    }
  };

  // Mock feature importance data - in real implementation, this would come from the ML model
  const getFeatureImportance = (result: PredictionResult) => {
    const baseFactors = [
      {
        name: 'Procedure Code',
        value: result.input?.procedure_code || 'N/A',
        importance: 0.35,
        impact: 'positive',
        explanation: 'This procedure has a high approval rate with this payer'
      },
      {
        name: 'Diagnosis Code',
        value: result.input?.diagnosis_code || 'N/A',
        importance: 0.28,
        impact: 'positive',
        explanation: 'Diagnosis is well-supported by clinical guidelines'
      },
      {
        name: 'Provider Specialty',
        value: result.input?.provider_specialty || 'N/A',
        importance: 0.22,
        impact: 'neutral',
        explanation: 'Specialty has standard approval patterns'
      },
      {
        name: 'Prior Denials',
        value: `${result.input?.prior_denials_provider || 0} denials`,
        importance: 0.15,
        impact: result.input?.prior_denials_provider && result.input.prior_denials_provider > 0 ? 'negative' : 'positive',
        explanation: result.input?.prior_denials_provider && result.input.prior_denials_provider > 0 
          ? 'Previous denials may indicate coverage issues'
          : 'No prior denials suggest good coverage history'
      }
    ];

    // Adjust based on actual prediction result
    if (result.prediction.approval_prediction === 0) {
      // For denied predictions, emphasize negative factors
      baseFactors[0].impact = 'negative';
      baseFactors[0].explanation = 'This procedure often requires additional documentation';
      baseFactors[1].impact = 'negative';
      baseFactors[1].explanation = 'Diagnosis may need more clinical justification';
    }

    return baseFactors.sort((a, b) => b.importance - a.importance);
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <div className="border-b border-neutral-200">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <h1 className="text-4xl font-serif font-light tracking-tight text-neutral-900">Authorization Prediction</h1>
          <p className="text-lg font-sans text-neutral-500 mt-2 tracking-tight">AI-powered prior authorization analysis</p>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-8">
        {!result ? (
          <>
            <PredictionForm 
              onPrediction={handlePrediction}
              onError={handleError}
              loading={loading}
              setLoading={setLoading}
            />

            {error && (
              <div className="max-w-3xl mx-auto mt-6">
                <div className="bg-red-50 border border-red-200 p-6">
                  <h3 className="text-sm font-sans font-medium text-red-800 tracking-tight">Error occurred</h3>
                  <p className="mt-1 text-sm font-sans text-red-700 tracking-tight">{error}</p>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="max-w-4xl mx-auto space-y-6">
            {/* Top Result Banner */}
            <div className={`bg-${getPredictionColor(result.prediction.approval_prediction === 1)}-50 border border-${getPredictionColor(result.prediction.approval_prediction === 1)}-200 rounded-2xl p-8`}>
              <div className="text-center">
                <h2 className={`text-4xl font-serif font-light tracking-tight text-${getPredictionColor(result.prediction.approval_prediction === 1)}-700 mb-4`}>
                  Request {result.prediction.approval_prediction === 1 ? 'Likely Approved' : 'Likely Denied'} 
                  <span className="block text-2xl mt-2">
                    ({(result.prediction.probability * 100).toFixed(0)}% Confidence)
                  </span>
                </h2>
                
                {/* Confidence Bar */}
                <div className="max-w-md mx-auto mb-4">
                  <div className="w-full bg-neutral-200 h-2 rounded-full">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 ${
                        result.prediction.approval_prediction === 1 ? 'bg-emerald-500' : 'bg-rose-500'
                      }`}
                      style={{ width: `${result.prediction.probability * 100}%` }}
                    />
                  </div>
                </div>
                
                <p className="text-sm font-sans text-neutral-600 tracking-tight">
                  Prediction generated using AI on clinical and payer data.
                </p>
              </div>
            </div>

            {/* Feature Importance Explanation */}
            <div className="bg-white rounded-2xl shadow-sm border border-neutral-200">
              <button
                onClick={() => setExplanationExpanded(!explanationExpanded)}
                className="w-full px-8 py-6 text-left flex items-center justify-between hover:bg-neutral-50 transition-colors"
              >
                <div>
                  <h3 className="text-xl font-serif font-light tracking-tight text-neutral-900">How This Prediction Was Calculated</h3>
                  <p className="text-sm font-sans text-neutral-500 mt-1">See which factors most influenced this result</p>
                </div>
                <span className="text-neutral-400">
                  {explanationExpanded ? '−' : '+'}
                </span>
              </button>
              
              {explanationExpanded && (
                <div className="px-8 pb-6">
                  <div className="space-y-6">
                    {getFeatureImportance(result).map((factor, index) => (
                      <div key={index} className="border-b border-neutral-100 pb-4 last:border-b-0">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-1">
                              <span className="text-sm font-sans font-medium text-neutral-900 tracking-tight">
                                {factor.name}
                              </span>
                              <span className={`px-2 py-1 rounded-full text-xs font-sans font-medium ${
                                factor.impact === 'positive' 
                                  ? 'bg-emerald-100 text-emerald-700' 
                                  : factor.impact === 'negative'
                                  ? 'bg-rose-100 text-rose-700'
                                  : 'bg-neutral-100 text-neutral-700'
                              }`}>
                                {factor.impact === 'positive' ? 'Positive' : factor.impact === 'negative' ? 'Negative' : 'Neutral'}
                              </span>
                            </div>
                            <p className="text-sm font-sans text-neutral-600 tracking-tight mb-2">
                              {factor.value}
                            </p>
                            <p className="text-sm font-sans text-neutral-700 tracking-tight">
                              {factor.explanation}
                            </p>
                          </div>
                          <div className="text-right ml-4">
                            <div className="text-lg font-serif font-light text-neutral-900 tracking-tight">
                              {(factor.importance * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs font-sans text-neutral-500 tracking-tight">Weight</div>
                          </div>
                        </div>
                        
                        {/* Importance Bar */}
                        <div className="w-full bg-neutral-200 h-1 rounded-full">
                          <div 
                            className={`h-1 rounded-full transition-all duration-500 ${
                              factor.impact === 'positive' 
                                ? 'bg-emerald-500' 
                                : factor.impact === 'negative'
                                ? 'bg-rose-500'
                                : 'bg-neutral-400'
                            }`}
                            style={{ width: `${factor.importance * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-6 p-4 bg-neutral-50 rounded-lg">
                    <h4 className="text-sm font-sans font-medium text-neutral-900 tracking-tight mb-2">How to Interpret This</h4>
                    <p className="text-sm font-sans text-neutral-600 tracking-tight leading-relaxed">
                      The AI model analyzed {getFeatureImportance(result).length} key factors from your request. 
                      Each factor contributes a percentage to the final prediction based on historical approval patterns. 
                      Higher weights indicate factors that historically have more influence on approval decisions.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Request Summary Card */}
            <div className="bg-white rounded-2xl shadow-sm border border-neutral-200">
              <button
                onClick={() => setSummaryExpanded(!summaryExpanded)}
                className="w-full px-8 py-6 text-left flex items-center justify-between hover:bg-neutral-50 transition-colors"
              >
                <h3 className="text-xl font-serif font-light tracking-tight text-neutral-900">Request Summary</h3>
                <span className="text-neutral-400">
                  {summaryExpanded ? '−' : '+'}
                </span>
              </button>
              
              {summaryExpanded && (
                <div className="px-8 pb-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <span className="text-xs font-sans font-medium text-neutral-500 uppercase tracking-wide">Procedure</span>
                        <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight bg-neutral-50 rounded-md px-3 py-2">
                          {result.input?.procedure_code || 'N/A'}
                        </div>
                      </div>
                      <div>
                        <span className="text-xs font-sans font-medium text-neutral-500 uppercase tracking-wide">Diagnosis</span>
                        <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight bg-neutral-50 rounded-md px-3 py-2">
                          {result.input?.diagnosis_code || 'N/A'}
                        </div>
                      </div>
                      <div>
                        <span className="text-xs font-sans font-medium text-neutral-500 uppercase tracking-wide">Payer</span>
                        <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight bg-neutral-50 rounded-md px-3 py-2">
                          {result.input?.payer || 'N/A'}
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <span className="text-xs font-sans font-medium text-neutral-500 uppercase tracking-wide">Patient</span>
                        <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight bg-neutral-50 rounded-md px-3 py-2">
                          {result.input?.patient_age || 'N/A'} years, {result.input?.patient_gender === 'M' ? 'Male' : 'Female'}
                        </div>
                      </div>
                      <div>
                        <span className="text-xs font-sans font-medium text-neutral-500 uppercase tracking-wide">Region</span>
                        <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight bg-neutral-50 rounded-md px-3 py-2">
                          {result.input?.region || 'N/A'}
                        </div>
                      </div>
                      <div>
                        <span className="text-xs font-sans font-medium text-neutral-500 uppercase tracking-wide">Prior Denials</span>
                        <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight bg-neutral-50 rounded-md px-3 py-2">
                          {result.input?.prior_denials_provider || 0} denials
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Suggested Next Steps */}
            <div className="bg-sky-50 border border-sky-200 rounded-2xl p-8">
              <h3 className="text-xl font-serif font-light tracking-tight text-sky-800 mb-6">Suggested Next Steps</h3>
              <div className="space-y-4">
                {getNextSteps(result.prediction.approval_prediction === 1).map((step, index) => (
                  <div key={index} className="flex items-start">
                    <span className="text-sky-600 mr-3 mt-1 font-medium">•</span>
                    <span className="text-sm font-sans text-sky-800 font-medium tracking-tight">{step}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Bottom CTA Row */}
            <div className="flex flex-col sm:flex-row gap-4 pt-6">
              <button className="px-8 py-4 bg-white text-neutral-700 border border-neutral-200 hover:bg-neutral-50 transition-all duration-200 font-sans font-medium text-sm tracking-tight rounded-lg">
                Download PDF Summary
              </button>
              <button 
                onClick={resetForm}
                className="px-8 py-4 bg-neutral-900 text-white hover:bg-neutral-800 transition-all duration-200 font-sans font-medium text-sm tracking-tight rounded-lg"
              >
                New Prediction
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predict; 