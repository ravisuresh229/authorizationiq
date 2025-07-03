import React from 'react';

const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-6">About AuthorizationIQ</h1>
      
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">What is Prior Authorization?</h2>
        <p className="text-gray-700 mb-4">
          Prior Authorization (PA) is a requirement by insurance companies for certain medical procedures, 
          tests, or medications to be pre-approved before they can be performed or prescribed. This process 
          can be time-consuming and often delays patient care.
        </p>
      </section>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">How AuthorizationIQ Works</h2>
        <p className="text-gray-700 mb-4">
          Our ML-powered system analyzes historical authorization data to predict the likelihood of approval 
          for a given procedure. By inputting the CPT code, diagnosis code, insurance payer, and other 
          relevant information, healthcare providers can instantly see:
        </p>
        <ul className="list-disc ml-6 text-gray-700">
          <li>Approval probability percentage</li>
          <li>Key factors affecting the decision</li>
          <li>Recommendations to improve approval chances</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Technology Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-lg mb-2">Frontend</h3>
            <ul className="text-gray-700 space-y-1">
              <li>• React with TypeScript</li>
              <li>• Tailwind CSS</li>
              <li>• Deployed on Vercel</li>
            </ul>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-lg mb-2">Backend</h3>
            <ul className="text-gray-700 space-y-1">
              <li>• FastAPI (Python)</li>
              <li>• Scikit-learn ML Model</li>
              <li>• AWS S3 for model storage</li>
              <li>• Deployed on Railway</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Benefits</h2>
        <ul className="list-disc ml-6 text-gray-700 space-y-2">
          <li>Reduces administrative burden on healthcare staff</li>
          <li>Speeds up the authorization process</li>
          <li>Improves patient care by reducing delays</li>
          <li>Helps identify potential issues before submission</li>
        </ul>
      </section>
    </div>
  );
};

export default About; 