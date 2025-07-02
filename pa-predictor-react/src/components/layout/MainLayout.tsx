import React from 'react';
import Sidebar from './Sidebar';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <div className="p-6 min-h-full">
          {children}
        </div>
      </main>
    </div>
  );
};

export default MainLayout; 