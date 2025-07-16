import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      {/* Top Navigation */}
      <header className="border-b border-neutral-200">
        <div className="max-w-7xl mx-auto px-6 py-3 flex justify-between items-center">
          <Link to="/" className="text-lg font-serif font-light tracking-tight text-neutral-900">
            AuthorizationIQ
          </Link>
          <nav className="flex space-x-8 text-sm font-sans font-medium text-neutral-600 tracking-tight">
            <Link 
              to="/about" 
              className={`transition-colors duration-200 ${
                location.pathname === "/about" ? "text-neutral-900" : "hover:text-neutral-900"
              }`}
            >
              How it Works
            </Link>
            <Link 
              to="/predict" 
              className={`transition-colors duration-200 ${
                location.pathname === "/predict" ? "text-neutral-900" : "hover:text-neutral-900"
              }`}
            >
              Predict
            </Link>
            <Link 
              to="/contact" 
              className={`transition-colors duration-200 ${
                location.pathname === "/contact" ? "text-neutral-900" : "hover:text-neutral-900"
              }`}
            >
              Contact
            </Link>
          </nav>
        </div>
      </header>

      {/* Content */}
      <main>
        {children}
      </main>
    </div>
  );
};

export default MainLayout; 