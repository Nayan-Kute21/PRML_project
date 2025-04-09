import { useNavigate } from 'react-router-dom';

const ResultsDisplay = ({ results, modelId }) => {
  const navigate = useNavigate();
  
  // Get model name based on ID
  const getModelName = (id) => {
    const modelNames = {
      'knn': 'K-Nearest Neighbors',
      'decisionTree': 'Decision Tree',
      'svm': 'Support Vector Machine',
      'randomForest': 'Random Forest',
      'ann': 'Artificial Neural Network',
      'cnn': 'Convolutional Neural Network'
    };
    
    return modelNames[id] || id;
  };
  
  const handleNewAnalysis = () => {
    navigate('/');
  };
  
  return (
    <div className="max-w-4xl mx-auto my-8">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-[#292966] p-6">
          <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
          <p className="text-[#CCCCFF] mt-2">
            Model used: {getModelName(modelId)}
          </p>
        </div>
        
        <div className="p-6">
          <div className="flex flex-col md:flex-row gap-8">
            {/* Image Display */}
            <div className="md:w-1/2">
              <div className="bg-[#F0F0FF] p-4 rounded-lg">
                <h3 className="text-lg font-medium text-[#292966] mb-3">Analyzed Image</h3>
                <div className="bg-white p-2 rounded border border-[#A3A3CC]">
                  {results.imageData && (
                    <img
                      src={results.imageData}
                      alt="Analyzed leaf"
                      className="w-full h-auto rounded"
                    />
                  )}
                </div>
                <p className="text-sm text-[#5C5C99] mt-2">
                  Processing time: {results.processingTime}
                </p>
              </div>
            </div>
            
            {/* Results Display */}
            <div className="md:w-1/2">
              <div className="bg-[#F0F0FF] p-4 rounded-lg h-full">
                <h3 className="text-lg font-medium text-[#292966] mb-3">Prediction Results</h3>
                
                {results.predictions && results.predictions.length > 0 ? (
                  <div className="space-y-4">
                    {results.predictions.map((prediction, index) => (
                      <div 
                        key={index}
                        className={`p-3 rounded-lg ${index === 0 ? 'bg-[#CCCCFF]' : 'bg-white border border-[#A3A3CC]'}`}
                      >
                        <div className="flex justify-between items-center">
                          <span className={`font-bold ${index === 0 ? 'text-[#292966]' : 'text-[#5C5C99]'}`}>
                            {prediction.class}
                          </span>
                          <span className={`font-medium ${index === 0 ? 'text-[#292966]' : 'text-[#5C5C99]'}`}>
                            {(prediction.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-2 w-full bg-white rounded-full h-2.5 overflow-hidden">
                          <div 
                            className="h-2.5 rounded-full bg-[#292966]" 
                            style={{ width: `${prediction.probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-[#5C5C99]">No predictions available.</p>
                )}
              </div>
            </div>
          </div>
          
          <div className="mt-8 flex justify-center">
            <button
              onClick={handleNewAnalysis}
              className="px-6 py-3 bg-[#292966] text-white rounded-lg hover:bg-[#5C5C99] transition-colors"
            >
              Analyze Another Image
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;