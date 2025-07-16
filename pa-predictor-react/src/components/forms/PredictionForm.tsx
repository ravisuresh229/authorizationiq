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
          newErrors.patient_age = 'Age must be between 18 and 90';
        }
        if (!formData.patient_gender) {
          newErrors.patient_gender = 'Gender is required';
        }
        break;
      case 2:
        if (!formData.procedure_code) {
          newErrors.procedure_code = 'Procedure code is required';
        }
        if (!formData.diagnosis_code) {
          newErrors.diagnosis_code = 'Diagnosis code is required';
        }
        if (!formData.provider_specialty) {
          newErrors.provider_specialty = 'Specialty is required';
        }
        break;
      case 3:
        if (!formData.payer) {
          newErrors.payer = 'Payer is required';
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
      setCurrentStep(prev => Math.min(prev + 1, 3));
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
      onPrediction({ ...result, input: formData });
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const steps = [
    { number: 1, title: 'Patient Information', subtitle: 'Basic demographics' },
    { number: 2, title: 'Medical Details', subtitle: 'Procedure and diagnosis' },
    { number: 3, title: 'Insurance & Request', subtitle: 'Payer and additional info' }
  ];

  return (
    <div className="max-w-3xl mx-auto">
      {/* Progress Steps - Minimal Apple Style */}
      <div className="mb-12">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <div key={step.number} className="flex items-center flex-1">
              <div className="flex flex-col items-center">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-500 ${
                  currentStep > step.number 
                    ? 'bg-black text-white' 
                    : currentStep === step.number 
                    ? 'bg-black text-white scale-110' 
                    : 'bg-gray-100 text-gray-400'
                }`}>
                  {currentStep > step.number ? (
                    <span className="text-lg">✓</span>
                  ) : (
                    <span className="text-sm font-light">{step.number}</span>
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p className={`text-sm font-medium transition-all ${
                    currentStep >= step.number ? 'text-gray-900' : 'text-gray-400'
                  }`}>{step.title}</p>
                  <p className={`text-xs transition-all ${
                    currentStep >= step.number ? 'text-gray-500' : 'text-gray-300'
                  }`}>{step.subtitle}</p>
                </div>
              </div>
              {index < steps.length - 1 && (
                <div className={`flex-1 h-[1px] mx-8 transition-all duration-500 ${
                  currentStep > step.number ? 'bg-black' : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Form Content */}
      <div className="bg-white rounded-3xl p-10 shadow-sm">
        {/* Step 1: Patient Information */}
        {currentStep === 1 && (
          <div className="space-y-8 animate-fadeIn">
            <div>
              <h2 className="text-2xl font-light text-gray-900 mb-2 tracking-tight">Patient Information</h2>
              <p className="text-gray-600 text-[15px] leading-relaxed">Enter basic patient demographics</p>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-[15px] font-medium text-gray-700 mb-2 tracking-tight">Age</label>
                <input
                  type="number"
                  value={formData.patient_age || ''}
                  onChange={(e) => updateFormData('patient_age', parseInt(e.target.value) || 18)}
                  className={`w-full px-0 py-4 border-0 border-b-2 focus:border-blue-600 focus:outline-none text-2xl font-light transition-all ${
                    errors.patient_age ? 'border-red-300' : 'border-gray-200'
                  }`}
                  placeholder="Enter age"
                  min="18"
                  max="90"
                />
                {errors.patient_age && (
                  <p className="mt-2 text-sm text-red-600">{errors.patient_age}</p>
                )}
              </div>

              <div>
                <label className="block text-[15px] font-medium text-gray-700 mb-4 tracking-tight">Gender</label>
                <div className="flex space-x-4">
                  {[
                    { value: 'M', label: 'Male' },
                    { value: 'F', label: 'Female' }
                  ].map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => updateFormData('patient_gender', option.value as 'M' | 'F')}
                      className={`flex-1 py-4 px-6 rounded-xl border-2 transition-all shadow-sm hover:shadow-md ${
                        formData.patient_gender === option.value
                          ? 'border-blue-600 bg-blue-600 text-white'
                          : 'border-gray-200 hover:border-gray-400 hover:scale-[1.02]'
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                {errors.patient_gender && (
                  <p className="mt-2 text-sm text-red-600">{errors.patient_gender}</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Medical Details */}
        {currentStep === 2 && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-light text-gray-900 mb-2">Medical Details</h2>
              <p className="text-gray-500">Specify procedure and diagnosis information</p>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Procedure Code (CPT)</label>
                <Autocomplete
                  options={cptCodes.map(code => ({ 
                    value: code.code, 
                    label: `${code.code} - ${code.description}` 
                  }))}
                  onSelect={selected => {
                    const code = selected[0]?.value || '';
                    updateFormData('procedure_code', code);
                  }}
                  placeholder="Search CPT codes..."
                  multiple={false}
                />
                {errors.procedure_code && (
                  <p className="mt-2 text-sm text-red-600">{errors.procedure_code}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Diagnosis Code (ICD-10)</label>
                <Autocomplete
                  options={icd10Codes.map(code => ({ 
                    value: code.code, 
                    label: `${code.code} - ${code.description}` 
                  }))}
                  onSelect={selected => {
                    const code = selected[0]?.value || '';
                    updateFormData('diagnosis_code', code);
                  }}
                  placeholder="Search ICD-10 codes..."
                  multiple={false}
                />
                {errors.diagnosis_code && (
                  <p className="mt-2 text-sm text-red-600">{errors.diagnosis_code}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Provider Specialty</label>
                <select
                  value={formData.provider_specialty}
                  onChange={(e) => updateFormData('provider_specialty', e.target.value)}
                  className={`w-full px-0 py-4 border-0 border-b-2 focus:border-black focus:outline-none text-lg font-light appearance-none bg-transparent transition-all ${
                    errors.provider_specialty ? 'border-red-300' : 'border-gray-200'
                  }`}
                >
                  <option value="">Select specialty</option>
                  {specialties.map(specialty => (
                    <option key={specialty} value={specialty}>{specialty}</option>
                  ))}
                </select>
                {errors.provider_specialty && (
                  <p className="mt-2 text-sm text-red-600">{errors.provider_specialty}</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Insurance & Request Details */}
        {currentStep === 3 && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-light text-gray-900 mb-2">Insurance & Request Details</h2>
              <p className="text-gray-500">Complete the authorization request</p>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Insurance Payer</label>
                <select
                  value={formData.payer}
                  onChange={(e) => updateFormData('payer', e.target.value)}
                  className={`w-full px-0 py-4 border-0 border-b-2 focus:border-black focus:outline-none text-lg font-light appearance-none bg-transparent transition-all ${
                    errors.payer ? 'border-red-300' : 'border-gray-200'
                  }`}
                >
                  <option value="">Select payer</option>
                  {payers.map(payer => (
                    <option key={payer} value={payer}>{payer}</option>
                  ))}
                </select>
                {errors.payer && (
                  <p className="mt-2 text-sm text-red-600">{errors.payer}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-4">Urgent Request?</label>
                  <div className="flex space-x-2">
                    {[
                      { value: 'Y', label: 'Yes' },
                      { value: 'N', label: 'No' }
                    ].map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => updateFormData('urgency_flag', option.value as 'Y' | 'N')}
                        className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                          formData.urgency_flag === option.value
                            ? 'border-black bg-black text-white'
                            : 'border-gray-200 hover:border-gray-400'
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-4">Documentation Complete?</label>
                  <div className="flex space-x-2">
                    {[
                      { value: 'Y', label: 'Yes' },
                      { value: 'N', label: 'No' }
                    ].map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => updateFormData('documentation_complete', option.value as 'Y' | 'N')}
                        className={`flex-1 py-3 px-4 rounded-xl border-2 transition-all ${
                          formData.documentation_complete === option.value
                            ? 'border-black bg-black text-white'
                            : 'border-gray-200 hover:border-gray-400'
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Prior Denials</label>
                  <input
                    type="number"
                    value={formData.prior_denials_provider || ''}
                    onChange={(e) => updateFormData('prior_denials_provider', parseInt(e.target.value) || 0)}
                    className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light transition-all"
                    placeholder="0"
                    min="0"
                    max="10"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Region</label>
                  <select
                    value={formData.region}
                    onChange={(e) => updateFormData('region', e.target.value as 'Midwest' | 'Northeast' | 'South' | 'West')}
                    className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light appearance-none bg-transparent transition-all"
                  >
                    <option value="">Select region</option>
                    <option value="Midwest">Midwest</option>
                    <option value="Northeast">Northeast</option>
                    <option value="South">South</option>
                    <option value="West">West</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-12">
          {currentStep > 1 && (
            <button
              type="button"
              onClick={prevStep}
              className="px-8 py-4 text-sm font-medium text-gray-700 hover:text-black transition-all"
            >
              Back
            </button>
          )}

          {currentStep < 3 ? (
            <button
              type="button"
              onClick={nextStep}
              className="ml-auto px-8 py-4 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all flex items-center group shadow-md hover:shadow-lg hover:scale-[1.02]"
            >
              Continue
              <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className="ml-auto px-8 py-4 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center shadow-md hover:shadow-lg hover:scale-[1.02]"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  Get Prediction
                  <span className="ml-2">→</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionForm; 