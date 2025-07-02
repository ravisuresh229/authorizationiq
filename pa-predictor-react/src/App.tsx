import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Predict from './pages/Predict';
import RecentPredictions from './pages/RecentPredictions';
import MainLayout from './components/layout/MainLayout';
import About from './pages/About';
import './index.css';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/recent" element={<RecentPredictions />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App; 