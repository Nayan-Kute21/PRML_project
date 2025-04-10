import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import ModelSelection from './components/ModelSelection';
import ImageUpload from './components/iamgeupload';
import ResultsDisplay from './components/ResultsDisplay';

// Create a separate header component to use useNavigate hook
function Header() {
  const navigate = useNavigate();
  
  return (
    <header className="py-4 bg-[#292966] shadow-md">
      <div className="container mx-auto px-4">
        <h1 className="text-2xl font-bold text-white">
          <div 
            onClick={() => navigate('/')} 
            className="cursor-pointer"
          >
            Leaf Classification System
          </div>
        </h1>
      </div>
    </header>
  );
}

// Main App Component
function App() {
  const [selectedModel, setSelectedModel] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-[#CCCCFF] to-[#A3A3CC]">
        <Routes>
          <Route path="*" element={<Header />} />
        </Routes>

        <main className="container mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<ModelSelection onModelSelect={setSelectedModel} />} />
            <Route
              path="/upload"
              element={
                selectedModel ?
                  <ImageUpload
                    modelId={selectedModel}
                    setAnalysisResults={setAnalysisResults}
                  /> :
                  <Navigate to="/" replace />
              }
            />
            <Route
              path="/results"
              element={
                analysisResults ?
                  <ResultsDisplay
                    results={analysisResults}
                    modelId={selectedModel}
                  /> :
                  <Navigate to="/" replace />
              }
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;