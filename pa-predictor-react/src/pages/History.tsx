import React from 'react';

const History: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="px-8 py-8">
          <h1 className="text-4xl font-light text-gray-900">Prediction History</h1>
          <p className="text-lg text-gray-500 mt-2">Track and analyze your authorization predictions</p>
        </div>
      </div>

      {/* Content */}
      <div className="px-8 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center">
            <h2 className="text-2xl font-light text-gray-900 mb-4">
              History Coming Soon
            </h2>
            <p className="text-lg text-gray-600">
              This will show your prediction history with search and filtering capabilities.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default History; 