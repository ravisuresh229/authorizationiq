import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';

const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: 'Home' },
    { name: 'Predict', href: '/predict', icon: 'Predict' },
    { name: 'About', href: '/about', icon: 'Info' },
  ];

  return (
    <aside className={`${collapsed ? 'w-20' : 'w-72'} h-screen bg-white/70 backdrop-blur-sm shadow-sm border-r border-gray-200 text-gray-800 transition-all duration-500 ease-in-out flex flex-col`}>
      {/* Header */}
      <div className="h-20 flex items-center justify-between px-6 border-b border-gray-200">
        {!collapsed && (
          <div>
            <h1 className="text-xl font-light tracking-tight text-gray-900">AuthorizationIQ</h1>
            <p className="text-xs text-gray-500 mt-0.5">Enterprise Edition</p>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 hover:bg-gray-100 rounded-lg transition-all ml-auto text-gray-600 hover:text-gray-900"
        >
          {collapsed ? 'U' : '←'}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 group relative ${
                    isActive 
                      ? 'bg-blue-50 text-blue-900 border-l-4 border-blue-600' 
                      : 'hover:bg-gray-100 text-gray-700 hover:text-gray-900'
                  }`}
                >
                  <span className={`text-xs font-medium ${collapsed ? 'mx-auto' : 'mr-3'} ${
                    isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-700'
                  }`}>
                    {item.icon.charAt(0)}
                  </span>
                  {!collapsed && (
                    <span className="text-sm font-medium">{item.name}</span>
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* User Section */}
      <div className="p-6 border-t border-gray-200">
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center">
            <span className="text-sm font-medium text-white">JD</span>
          </div>
          {!collapsed && (
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-900">Dr. Johnson</p>
              <p className="text-xs text-gray-500">Cardiology</p>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar; 