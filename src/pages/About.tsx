import React, { useEffect, useState } from 'react';
import { api } from '../services/api';

const About: React.FC = () => {
  const [about, setAbout] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Debug: log the API URL being used
    console.log('API URL:', process.env.REACT_APP_API_URL);
    api.getAbout()
      .then(setAbout)
      .catch(() => setError('Failed to load about info'));
  }, []);

  if (error) return <div className="text-red-600">{error}</div>;
  if (!about) return <div>Loading...</div>;

  return (
    <div className="max-w-2xl mx-auto p-8 bg-white rounded shadow">
      <h1 className="text-2xl font-bold mb-4">AuthorizationIQ</h1>
      <p className="mb-4">AuthorizationIQ is an AI-powered tool for predicting prior authorization requirements and approval likelihood for medical procedures. Enter patient, procedure, and payer details to receive instant, data-driven insights and recommendations. Built for healthcare professionals who want to streamline the authorization process and reduce denials.</p>
      <h2 className="text-xl font-semibold mb-2">Key Features</h2>
      <ul className="list-disc pl-6">
        {about.features.map((f: string, i: number) => <li key={i}>{f}</li>)}
      </ul>
    </div>
  );
};

export default About; 