import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ModelButton from './ModelButton';

const ModelSelection = ({ onModelSelect }) => {
  const [selectedModel, setSelectedModel] = useState(null);
  const navigate = useNavigate();
  
  // Sample model data
  const models = [
    {
      id: 'KNN',
      name: 'K-Nearest Neighbors',
      description: 'A simple, instance-based learning algorithm that classifies new data points based on similarity measures.',
      stats: {
        'Accuracy': 0.89,
        'F1 Score': 0.87,
      }
    },
    {
      id: 'DT',
      name: 'Decision Tree',
      description: 'A tree-like model of decisions where leaves represent class labels and branches represent feature conditions.',
      stats: {
        'Accuracy': 0.85,
        'F1 Score': 0.83,
      }
    },
    {
      id: 'SVM',
      name: 'Support Vector Machine',
      description: 'Finds the hyperplane that best separates classes in a high-dimensional feature space.',
      stats: {
        'Accuracy': 0.92,
        'F1 Score': 0.91,
      }
    },
    {
      id: 'RF',
      name: 'Random Forest',
      description: 'An ensemble learning method that constructs multiple decision trees during training.',
      stats: {
        'Accuracy': 0.94,
        'F1 Score': 0.93,
      }
    },
    {
      id: 'ANN',
      name: 'Artificial Neural Network',
      description: 'A computational model inspired by the human brain that learns patterns from input data.',
      stats: {
        'Accuracy': 0.91,
        'F1 Score': 0.90,
      } 
    },
    
  ];
  
  const handleModelSelect = (modelId) => {
    setSelectedModel(modelId);
    if (onModelSelect) {
      onModelSelect(modelId);
    }
  };
  
  const handleContinue = () => {
    navigate('/upload');
  };
  
  return (
    <div className="py-8">
      <h1 className="text-3xl font-bold text-[#292966] mb-6 text-center">
        Leaf Classification Models
      </h1>
      <p className="text-[#5C5C99] text-lg mb-8 text-center max-w-3xl mx-auto">
        Select a model to classify leaf images. Each model has different strengths and 
        performance characteristics for identifying leaf species.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((model) => (
          <ModelButton
            key={model.id}
            model={model.id}
            name={model.name}
            description={model.description}
            stats={model.stats}
            isSelected={selectedModel === model.id}
            onClick={() => handleModelSelect(model.id)}
          />
        ))}
      </div>
      
      {selectedModel && (
        <div className="mt-8 p-4 bg-white rounded-lg shadow-lg text-center">
          <p className="text-[#5C5C99]">
            Selected model: <span className="font-bold text-[#292966]">
              {models.find(m => m.id === selectedModel)?.name}
            </span>
          </p>
          <button 
            className="mt-4 px-6 py-3 bg-[#292966] text-white rounded-lg hover:bg-[#5C5C99] transition-colors font-medium"
            onClick={handleContinue}
          >
            Continue to Image Upload
          </button>
        </div>
      )}
    </div>
  );
};

export default ModelSelection;