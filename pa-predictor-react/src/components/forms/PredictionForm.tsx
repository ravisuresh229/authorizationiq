import React, { useState, useEffect } from 'react';
import { PredictionInput, PredictionResult, CPTCode, ICD10Code } from '../../types/prediction';
import { api } from '../../services/api';
import { dataService } from '../../services/dataService';
import Autocomplete from '../ui/Autocomplete';

interface PredictionFormProps {
  onPrediction: (result: PredictionResult) => void;
  onError: (error: string) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPrediction,
  onError,
  loading,
  setLoading
}) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [cptCodes, setCptCodes] = useState<CPTCode[]>([]);
  const [icd10Codes, setIcd10Codes] = useState<ICD10Code[]>([]);
  const [specialties, setSpecialties] = useState<string[]>([]);
  const [payers, setPayers] = useState<string[]>([]);
  const [formData, setFormData] = useState<PredictionInput>({
    patient_age: 18,
    patient_gender: 'M',
    procedure_code: '',
    diagnosis_code: '',
    provider_specialty: '',
    payer: '',
    urgency_flag: 'N',
    documentation_complete: 'Y',
    prior_denials_provider: 0,
    region: 'South'
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    const loadData = async () => {
      try {
        const [cptData, icd10Data, specialtyData, payerData] = await Promise.all([
          dataService.loadCPTCodes(),
          dataService.loadICD10Codes(),
          dataService.loadSpecialties(),
          dataService.loadPayers()
        ]);
        setCptCodes(cptData);
        setIcd10Codes(icd10Data);
        setSpecialties(specialtyData);
        setPayers(payerData);
      } catch (error) {
        console.error('Error loading data:', error);
        onError('Failed to load form data. Please refresh the page.');
      }
    };
    loadData();
  }, [onError]);

  const updateFormData = (field: keyof PredictionInput, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const validateStep = (step: number): boolean => {
    const newErrors: Record<string, string> = {};

    switch (step) {
      case 1:
        if (!formData.patient_age || formData.patient_age < 18 || formData.patient_age > 90) {
          newErrors.patient_age = 'Patient age must be between 18 and 90';
        }
        if (!formData.patient_gender) {
          newErrors.patient_gender = 'Patient gender is required';
        }
        if (!formData.procedure_code) {
          newErrors.procedure_code = 'Procedure code is required';
        } else if (!cptCodes.some(c => c.code === formData.procedure_code.toUpperCase())) {
          newErrors.procedure_code = 'Please enter a valid CPT code';
        }
        if (!formData.diagnosis_code) {
          newErrors.diagnosis_code = 'Diagnosis code is required';
        } else if (!icd10Codes.some(c => c.code === formData.diagnosis_code.toUpperCase())) {
          newErrors.diagnosis_code = 'Please enter a valid ICD-10 code';
        }
        if (!formData.provider_specialty) {
          newErrors.provider_specialty = 'Provider specialty is required';
        }
        break;
      case 2:
        if (!formData.payer) {
          newErrors.payer = 'Payer is required';
        }
        if (!formData.urgency_flag) {
          newErrors.urgency_flag = 'Urgency flag is required';
        }
        if (!formData.documentation_complete) {
          newErrors.documentation_complete = 'Documentation complete is required';
        }
        if (formData.prior_denials_provider === undefined || formData.prior_denials_provider < 0) {
          newErrors.prior_denials_provider = 'Prior denials is required';
        }
        if (!formData.region) {
          newErrors.region = 'Region is required';
        }
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const nextStep = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, 2));
    }
  };

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;

    setLoading(true);
    try {
      const result = await api.predict(formData);
      onPrediction(result);
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const steps = [
    { number: 1, title: 'Patient & Procedure Information' },
    { number: 2, title: 'Request Details' }
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      {/* Stepper */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <div key={step.number} className="flex items-center">
              <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                currentStep >= step.number
                  ? 'bg-blue-600 border-blue-600 text-white'
                  : 'border-gray-300 text-gray-500'
              }`}>
                {step.number}
              </div>
              <span className={`ml-2 text-sm font-medium ${
                currentStep >= step.number ? 'text-blue-600' : 'text-gray-500'
              }`}>
                {step.title}
              </span>
              {index < steps.length - 1 && (
                <div className={`w-16 h-0.5 mx-4 ${
                  currentStep > step.number ? 'bg-blue-600' : 'bg-gray-300'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: Patient & Procedure Information */}
      {currentStep === 1 && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-900">Patient & Procedure Information</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Patient Age
              </label>
              <input
                type="number"
                value={formData.patient_age || ''}
                onChange={(e) => updateFormData('patient_age', parseInt(e.target.value) || 18)}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.patient_age ? 'border-red-300' : 'border-gray-300'
                }`}
                placeholder="Enter patient age"
                min="18"
                max="90"
              />
              {errors.patient_age && (
                <p className="mt-1 text-sm text-red-600">{errors.patient_age}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Patient Gender
              </label>
              <select
                value={formData.patient_gender}
                onChange={(e) => updateFormData('patient_gender', e.target.value as 'M' | 'F')}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.patient_gender ? 'border-red-300' : 'border-gray-300'
                }`}
              >
                <option value="">Select gender</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
              {errors.patient_gender && (
                <p className="mt-1 text-sm text-red-600">{errors.patient_gender}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Procedure Code (CPT)
              </label>
              <Autocomplete
                options={cptCodes.map(code => ({ value: code.code, label: `${code.code} - ${code.description}` }))}
                onSelect={selected => {
                  const code = selected[0]?.value || '';
                  updateFormData('procedure_code', code);
                }}
                placeholder="Type to search CPT codes..."
                multiple={false}
              />
              {errors.procedure_code && (
                <p className="mt-1 text-sm text-red-600">{errors.procedure_code}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Diagnosis Code (ICD-10)
              </label>
              <Autocomplete
                options={icd10Codes.map(code => ({ value: code.code, label: `${code.code} - ${code.description}` }))}
                onSelect={selected => {
                  const code = selected[0]?.value || '';
                  updateFormData('diagnosis_code', code);
                }}
                placeholder="Type to search ICD-10 codes..."
                multiple={false}
              />
              {errors.diagnosis_code && (
                <p className="mt-1 text-sm text-red-600">{errors.diagnosis_code}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Provider Specialty
              </label>
              <select
                value={formData.provider_specialty}
                onChange={(e) => updateFormData('provider_specialty', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.provider_specialty ? 'border-red-300' : 'border-gray-300'
                }`}
              >
                <option value="">Select specialty</option>
                {specialties.map(specialty => (
                  <option key={specialty} value={specialty}>{specialty}</option>
                ))}
              </select>
              {errors.provider_specialty && (
                <p className="mt-1 text-sm text-red-600">{errors.provider_specialty}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Request Details */}
      {currentStep === 2 && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-900">Request Details</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Insurance Payer
              </label>
              <select
                value={formData.payer}
                onChange={(e) => updateFormData('payer', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.payer ? 'border-red-300' : 'border-gray-300'
                }`}
              >
                <option value="">Select payer</option>
                {payers.map(payer => (
                  <option key={payer} value={payer}>{payer}</option>
                ))}
              </select>
              {errors.payer && (
                <p className="mt-1 text-sm text-red-600">{errors.payer}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Urgent Request
              </label>
              <select
                value={formData.urgency_flag}
                onChange={(e) => updateFormData('urgency_flag', e.target.value as 'Y' | 'N')}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.urgency_flag ? 'border-red-300' : 'border-gray-300'
                }`}
              >
                <option value="">Select urgency</option>
                <option value="Y">Yes</option>
                <option value="N">No</option>
              </select>
              {errors.urgency_flag && (
                <p className="mt-1 text-sm text-red-600">{errors.urgency_flag}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Documentation Complete
              </label>
              <select
                value={formData.documentation_complete}
                onChange={(e) => updateFormData('documentation_complete', e.target.value as 'Y' | 'N')}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.documentation_complete ? 'border-red-300' : 'border-gray-300'
                }`}
              >
                <option value="">Select status</option>
                <option value="Y">Yes</option>
                <option value="N">No</option>
              </select>
              {errors.documentation_complete && (
                <p className="mt-1 text-sm text-red-600">{errors.documentation_complete}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Prior Denials
              </label>
              <input
                type="number"
                value={formData.prior_denials_provider || ''}
                onChange={(e) => updateFormData('prior_denials_provider', parseInt(e.target.value) || 0)}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.prior_denials_provider ? 'border-red-300' : 'border-gray-300'
                }`}
                placeholder="Enter number of prior denials"
                min="0"
                max="10"
              />
              {errors.prior_denials_provider && (
                <p className="mt-1 text-sm text-red-600">{errors.prior_denials_provider}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Region
              </label>
              <select
                value={formData.region}
                onChange={(e) => updateFormData('region', e.target.value as 'Midwest' | 'Northeast' | 'South' | 'West')}
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                  errors.region ? 'border-red-300' : 'border-gray-300'
                }`}
              >
                <option value="">Select region</option>
                <option value="Midwest">Midwest</option>
                <option value="Northeast">Northeast</option>
                <option value="South">South</option>
                <option value="West">West</option>
              </select>
              {errors.region && (
                <p className="mt-1 text-sm text-red-600">{errors.region}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Navigation Buttons */}
      <div className="flex justify-between mt-8">
        <button
          type="button"
          onClick={prevStep}
          disabled={currentStep === 1}
          className="px-6 py-3 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-sm hover:shadow-md"
        >
          ← Previous
        </button>

        {currentStep < 2 ? (
          <button
            type="button"
            onClick={nextStep}
            className="px-6 py-3 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-lg hover:bg-blue-700 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 shadow-sm"
          >
            Next →
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading}
            className="px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-green-600 to-green-700 border border-transparent rounded-lg hover:from-green-700 hover:to-green-800 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-sm"
          >
            {loading ? (
              <div className="flex items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </div>
            ) : (
              '🚀 Predict Approval'
            )}
          </button>
        )}
      </div>
    </div>
  );
};

export default PredictionForm; 