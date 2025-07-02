import React from 'react';
import { useLocation } from 'react-router-dom';
import { FaChartBar, FaClock, FaInfoCircle, FaUserCircle, FaChevronLeft } from 'react-icons/fa';
import { useState } from 'react';

const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);

  const ChartBar = FaChartBar as React.ElementType;
  const Clock = FaClock as React.ElementType;
  const InfoCircle = FaInfoCircle as React.ElementType;
  const UserCircle = FaUserCircle as React.ElementType;
  const ChevronLeft = FaChevronLeft as React.ElementType;

  return (
    <aside className={`h-screen transition-all duration-300 ${collapsed ? 'w-20' : 'w-64'} bg-white/80 backdrop-blur-lg shadow-xl relative border-r border-gray-200`}>
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
        <span className="font-bold text-xl text-gray-900 transition-all duration-300">
          {collapsed ? 'AIQ' : 'AuthorizationIQ'}
        </span>
        <button 
          onClick={() => setCollapsed(!collapsed)} 
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <ChevronLeft className={`w-4 h-4 text-gray-600 transition-transform ${collapsed ? 'rotate-180' : ''}`} />
        </button>
      </div>
      
      <nav className="mt-6 flex flex-col gap-2 px-3">
        <SidebarItem 
          icon={<ChartBar />} 
          label="Predict" 
          to="/predict" 
          tooltip="Make Predictions" 
          collapsed={collapsed} 
        />
        <SidebarItem 
          icon={<Clock />} 
          label="Recent Predictions" 
          to="/recent" 
          tooltip="Recent Predictions" 
          collapsed={collapsed} 
        />
        <SidebarItem 
          icon={<InfoCircle />} 
          label="About" 
          to="/about" 
          tooltip="About" 
          collapsed={collapsed} 
        />
      </nav>
      
      <div className="absolute bottom-0 left-0 w-full px-6 py-4 border-t border-gray-200 flex items-center gap-3">
        <UserCircle className="text-2xl text-gray-400" />
        {!collapsed && <span className="text-sm text-gray-600">Admin User</span>}
      </div>
    </aside>
  );
};

interface SidebarItemProps {
  icon: React.ReactNode;
  label: string;
  to: string;
  tooltip: string;
  collapsed: boolean;
}

const SidebarItem: React.FC<SidebarItemProps> = ({ icon, label, to, tooltip, collapsed }) => (
  <a 
    href={to} 
    className="group flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-blue-50 hover:text-blue-700 transition-all duration-200 relative" 
    title={tooltip}
  >
    <span className="text-lg text-gray-600 group-hover:text-blue-600 group-hover:scale-110 transition-all duration-200">
      {icon}
    </span>
    {!collapsed && (
      <span className="text-sm font-medium text-gray-700 group-hover:text-blue-700 transition-colors">
        {label}
      </span>
    )}
  </a>
);

export default Sidebar; 