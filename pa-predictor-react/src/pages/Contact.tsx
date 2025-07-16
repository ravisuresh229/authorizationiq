import React from 'react';

const Contact: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-6 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-light text-gray-900 mb-4 tracking-tight">
            Contact Us
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Get in touch with our team to learn more about our PA prediction platform 
            and how it can transform your healthcare operations.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-16">
          {/* Contact Form */}
          <div className="bg-white rounded-3xl p-10 shadow-sm">
            <h2 className="text-2xl font-light text-gray-900 mb-8 tracking-tight">
              Send us a message
            </h2>
            
            <form className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 tracking-tight">
                    First Name
                  </label>
                  <input
                    type="text"
                    className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light transition-all"
                    placeholder="Enter first name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 tracking-tight">
                    Last Name
                  </label>
                  <input
                    type="text"
                    className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light transition-all"
                    placeholder="Enter last name"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2 tracking-tight">
                  Email Address
                </label>
                <input
                  type="email"
                  className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light transition-all"
                  placeholder="Enter email address"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2 tracking-tight">
                  Organization
                </label>
                <input
                  type="text"
                  className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light transition-all"
                  placeholder="Enter organization name"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2 tracking-tight">
                  Subject
                </label>
                <select className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light appearance-none bg-transparent transition-all">
                  <option value="">Select a subject</option>
                  <option value="demo">Request Demo</option>
                  <option value="pricing">Pricing Inquiry</option>
                  <option value="integration">Integration Support</option>
                  <option value="partnership">Partnership Opportunity</option>
                  <option value="general">General Inquiry</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2 tracking-tight">
                  Message
                </label>
                <textarea
                  rows={4}
                  className="w-full px-0 py-4 border-0 border-b-2 border-gray-200 focus:border-black focus:outline-none text-lg font-light transition-all resize-none"
                  placeholder="Tell us about your needs..."
                />
              </div>

              <button
                type="submit"
                className="w-full px-8 py-4 bg-black text-white rounded-xl hover:bg-gray-800 transition-all flex items-center justify-center group shadow-md hover:shadow-lg hover:scale-[1.02]"
              >
                Send Message
                <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
              </button>
            </form>
          </div>

          {/* Contact Information */}
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-light text-gray-900 mb-8 tracking-tight">
                Get in touch
              </h2>
              <p className="text-gray-600 leading-relaxed mb-8">
                Our team is here to help you understand how our AI-powered PA prediction 
                platform can streamline your prior authorization process and improve 
                patient outcomes.
              </p>
            </div>

            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <svg className="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-1">Email</h3>
                  <p className="text-gray-600">ravikirans723@gmail.com</p>
                  <p className="text-sm text-gray-500">We typically respond within 24 hours</p>
                </div>
              </div>
            </div>

            {/* Additional Info */}
            <div className="bg-gray-50 rounded-2xl p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Ready to get started?</h3>
              <p className="text-gray-600 text-sm leading-relaxed mb-4">
                Schedule a personalized demo to see our platform in action and learn 
                how it can benefit your organization.
              </p>
              <button className="w-full px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all text-sm font-medium">
                Schedule Demo
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact; 