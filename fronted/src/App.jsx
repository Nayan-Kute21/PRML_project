import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ImageUpload from './components/iamgeupload';
import ResultsDisplay from './components/ResultsDisplay';

// Create a separate header component
function Header() {
  return (
    <header className="py-4 bg-[#292966] shadow-md">
      <div className="container mx-auto px-4">
        <h1 className="text-2xl font-bold text-white">
          Leaf Classification System
        </h1>
      </div>
    </header>
  );
}

// Main App Component
function App() {
  const [analysisResults, setAnalysisResults] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-[#CCCCFF] to-[#A3A3CC]">
        <Header />

        <main className="container mx-auto px-4 py-6">
          <Routes>
            <Route 
              path="/" 
              element={<ImageUpload setAnalysisResults={setAnalysisResults} />} 
            />
            <Route
              path="/results"
              element={
                analysisResults ?
                  <ResultsDisplay
                    results={analysisResults}
                    modelId="SVM"
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