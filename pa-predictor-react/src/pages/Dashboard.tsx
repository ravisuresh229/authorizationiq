import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const Dashboard: React.FC = () => {
  const [currentPrediction, setCurrentPrediction] = useState(0);

  const mockPredictions = [
    {
      procedure: "CPT 99213",
      diagnosis: "ICD-10 Z51.11",
      specialty: "Internal Medicine",
      payer: "Blue Cross Blue Shield",
      result: "Likely Approved",
      confidence: "94%",
      color: "emerald"
    },
    {
      procedure: "CPT 73721",
      diagnosis: "ICD-10 M79.3",
      specialty: "Radiology",
      payer: "Aetna",
      result: "Likely Denied",
      confidence: "87%",
      color: "red"
    },
    {
      procedure: "CPT 99204",
      diagnosis: "ICD-10 I10",
      specialty: "Cardiology",
      payer: "UnitedHealth",
      result: "Likely Approved",
      confidence: "91%",
      color: "emerald"
    }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPrediction((prev) => (prev + 1) % mockPredictions.length);
    }, 6000);
    return () => clearInterval(interval);
  }, [mockPredictions.length]);

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Hero Section - Enhanced Split Layout */}
      <div className="max-w-7xl mx-auto px-6 py-24">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
          {/* Left Column - Content */}
          <div className="space-y-10">
            {/* Enhanced Headline */}
            <div className="space-y-6">
              <h1 className="text-6xl font-serif font-light tracking-tight text-neutral-900 leading-tight">
                Make Faster Decisions With AI You Can Trust.
              </h1>
              
              {/* Enhanced Subheadline */}
              <p className="text-xl font-sans text-neutral-600 leading-relaxed tracking-tight max-w-lg">
                AI-powered predictions trained on over 50,000 historical cases. Reduce denials, accelerate approvals, and make data-backed clinical decisions instantly.
              </p>
            </div>

            {/* Enhanced CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Link 
                to="/predict" 
                className="px-10 py-4 bg-black text-white hover:bg-neutral-800 transition-all duration-300 font-sans font-medium text-sm tracking-tight rounded-xl shadow-lg hover:shadow-xl hover:scale-[1.02] flex items-center justify-center group"
              >
                Run Prediction
                <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
              <Link 
                to="/about" 
                className="px-10 py-4 bg-white text-neutral-700 border-2 border-neutral-300 hover:border-neutral-400 hover:bg-neutral-50 transition-all duration-300 font-sans font-medium text-sm tracking-tight rounded-xl flex items-center justify-center group"
              >
                Learn More
                <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
          </div>

          {/* Right Column - Enhanced Live Prediction Card */}
          <div className="relative">
            <div className="bg-white rounded-3xl shadow-lg border border-neutral-200 p-10 relative overflow-hidden">
              {/* Subtle background tint */}
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-50/30 to-blue-50/30 pointer-events-none"></div>
              
              {/* Enhanced Header */}
              <div className="relative flex items-center justify-between mb-8">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></div>
                  <h3 className="text-xl font-sans font-semibold text-neutral-900 tracking-tight">Live Prediction</h3>
                </div>
                <div className="flex space-x-2">
                  {mockPredictions.map((_, index) => (
                    <div
                      key={index}
                      className={`w-3 h-3 rounded-full transition-all duration-500 ${
                        index === currentPrediction ? 'bg-emerald-500 scale-110' : 'bg-neutral-200'
                      }`}
                    />
                  ))}
                </div>
              </div>

              {/* Enhanced Prediction Content */}
              <div className="relative space-y-8">
                {mockPredictions.map((prediction, index) => (
                  <div
                    key={index}
                    className={`transition-all duration-700 ${
                      index === currentPrediction ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 absolute'
                    }`}
                    style={{ display: index === currentPrediction ? 'block' : 'none' }}
                  >
                    {/* Enhanced Prediction Details with Icons */}
                    <div className="grid grid-cols-2 gap-6 mb-8">
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                          <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div>
                          <span className="text-xs font-sans font-semibold text-neutral-500 uppercase tracking-wide">Procedure</span>
                          <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight font-medium">{prediction.procedure}</div>
                        </div>
                      </div>
                      
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                          <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                          </svg>
                        </div>
                        <div>
                          <span className="text-xs font-sans font-semibold text-neutral-500 uppercase tracking-wide">Diagnosis</span>
                          <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight font-medium">{prediction.diagnosis}</div>
                        </div>
                      </div>
                      
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                          <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                          </svg>
                        </div>
                        <div>
                          <span className="text-xs font-sans font-semibold text-neutral-500 uppercase tracking-wide">Specialty</span>
                          <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight font-medium">{prediction.specialty}</div>
                        </div>
                      </div>
                      
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center flex-shrink-0">
                          <svg className="w-4 h-4 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
                          </svg>
                        </div>
                        <div>
                          <span className="text-xs font-sans font-semibold text-neutral-500 uppercase tracking-wide">Payer</span>
                          <div className="mt-1 text-sm font-sans text-neutral-900 tracking-tight font-medium">{prediction.payer}</div>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Result with Badge */}
                    <div className="border-t border-neutral-100 pt-8">
                      <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center space-x-4">
                          <div className={`px-4 py-2 rounded-full flex items-center space-x-2 ${
                            prediction.color === 'emerald' 
                              ? 'bg-emerald-100 text-emerald-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {prediction.color === 'emerald' ? (
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                            ) : (
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                            )}
                            <span className="text-sm font-semibold">{prediction.result}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-4xl font-serif font-light tracking-tight ${
                            prediction.color === 'emerald' ? 'text-emerald-700' : 'text-red-700'
                          }`}>
                            {prediction.confidence}
                          </div>
                          <p className="text-sm font-sans text-neutral-500 tracking-tight">Confidence</p>
                        </div>
                      </div>
                      
                      {/* Enhanced Confidence Bar */}
                      <div className="mt-6">
                        <div className="w-full bg-neutral-200 h-2 rounded-full overflow-hidden">
                          <div 
                            className={`h-2 rounded-full transition-all duration-1000 ease-out ${
                              prediction.color === 'emerald' ? 'bg-emerald-500' : 'bg-red-500'
                            }`}
                            style={{ width: prediction.confidence }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced KPI Cards Section with Divider */}
      <div className="border-t border-neutral-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-neutral-50 rounded-2xl p-8 text-center hover:bg-white hover:shadow-lg transition-all duration-300 group">
              <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="text-3xl font-serif font-light text-neutral-900 mb-2 tracking-tight">94.8%</div>
              <div className="text-sm font-sans font-medium text-neutral-600 tracking-tight">Approval Accuracy</div>
            </div>
            
            <div className="bg-neutral-50 rounded-2xl p-8 text-center hover:bg-white hover:shadow-lg transition-all duration-300 group">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="text-3xl font-serif font-light text-neutral-900 mb-2 tracking-tight">&lt;2s</div>
              <div className="text-sm font-sans font-medium text-neutral-600 tracking-tight">Prediction Time</div>
            </div>
            
            <div className="bg-neutral-50 rounded-2xl p-8 text-center hover:bg-white hover:shadow-lg transition-all duration-300 group">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div className="text-3xl font-serif font-light text-neutral-900 mb-2 tracking-tight">50K+</div>
              <div className="text-sm font-sans font-medium text-neutral-600 tracking-tight">Cases Learned</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 