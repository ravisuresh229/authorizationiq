import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Predict from './pages/Predict';
import MainLayout from './components/layout/MainLayout';
import About from './pages/About';
import Dashboard from './pages/Dashboard';
import Contact from './pages/Contact';
import './index.css';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App; 