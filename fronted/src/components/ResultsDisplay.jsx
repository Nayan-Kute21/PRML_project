import { useNavigate } from 'react-router-dom';

const ResultsDisplay = ({ results, modelId }) => {
  const navigate = useNavigate();
  
  // Get model name based on ID
  const getModelName = (id) => {
    const modelNames = {
      'KNN': 'K-Nearest Neighbors',
      'DT': 'Decision Tree',
      'SVM': 'Support Vector Machine',
      'RF': 'Random Forest',
      'ANN': 'Artificial Neural Network',
      'CNN': 'Convolutional Neural Network'
    };
    
    return modelNames[id] || id;
  };
  
  const handleNewAnalysis = () => {
    navigate('/');
  };

  // Format the prediction label to be more readable
  const formatLabel = (label) => {
    if (!label) return '';
    
    // Replace underscores with spaces and capitalize each word
    return label.replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };
  
  return (
    <div className="max-w-4xl mx-auto my-8">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-[#292966] p-6">
          <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
          <p className="text-[#CCCCFF] mt-2">
            Model used: {getModelName(results.model_used || modelId)}
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
              </div>
            </div>
            
            {/* Results Display */}
            <div className="md:w-1/2">
              <div className="bg-[#F0F0FF] p-4 rounded-lg h-full">
                <h3 className="text-lg font-medium text-[#292966] mb-3">Prediction Results</h3>
                
                {results.prediction_label ? (
                  <div className="p-4 rounded-lg bg-[#CCCCFF]">
                    <div className="mb-2">
                      <span className="text-lg font-bold text-[#292966]">
                        Species Identified:
                      </span>
                    </div>
                    <div className="bg-white p-4 rounded-lg border border-[#A3A3CC]">
                      <span className="text-xl font-bold text-[#292966]">
                        {formatLabel(results.prediction_label)}
                      </span>
                      <div className="mt-2 text-[#5C5C99]">
                        <span className="font-medium">Class ID: </span>
                        {results.prediction_class}
                      </div>
                    </div>
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