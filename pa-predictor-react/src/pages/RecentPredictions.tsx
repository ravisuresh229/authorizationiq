import React, { useState, useEffect } from 'react';
import { RecentPrediction } from '../types/prediction';
import { api } from '../services/api';

const RecentPredictions: React.FC = () => {
  const [predictions, setPredictions] = useState<RecentPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const data = await api.getRecentPredictions();
        setPredictions(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading recent predictions...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-md p-4">
        <div className="flex">
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="mt-2 text-sm text-red-700">{error}</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Recent Predictions</h1>
        <p className="mt-2 text-gray-600">
          View your recent AuthorizationIQ predictions and their results.
        </p>
      </div>

      {predictions.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-500">No recent predictions found.</div>
        </div>
      ) : (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {predictions.map((prediction, index) => (
              <li key={index} className="px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-4">
                      <div className={`px-2 py-1 text-xs font-medium rounded-full ${
                        prediction.prediction === 'APPROVED' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {prediction.prediction}
                      </div>
                      <div className="text-sm text-gray-500">
                        {(prediction.confidence * 100).toFixed(1)}% confidence
                      </div>
                    </div>
                    
                    <div className="mt-2 text-sm text-gray-900">
                      <div><strong>Patient Age:</strong> {prediction.input.patient_age}</div>
                      <div><strong>Gender:</strong> {prediction.input.patient_gender}</div>
                      <div><strong>Payer:</strong> {prediction.input.payer}</div>
                      <div><strong>Specialty:</strong> {prediction.input.provider_specialty}</div>
                    </div>
                  </div>
                  
                  <div className="text-sm text-gray-500">
                    {new Date(prediction.timestamp).toLocaleString()}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default RecentPredictions; 