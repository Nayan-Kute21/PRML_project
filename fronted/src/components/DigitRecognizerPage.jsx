import { useState } from 'react';
import DigitRecognitionCanvas from './DigitRecognitionCanvas';

const DigitRecognizerPage = () => {
  const [selectedModel, setSelectedModel] = useState('cnn');
  
  const models = [
    { id: 'cnn', name: 'Convolutional Neural Network' },
    { id: 'mlp', name: 'Multi-Layer Perceptron' },
    { id: 'svm', name: 'Support Vector Machine' }
  ];
  
  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8 font-arial">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-black mb-2">
            Handwritten Digit Recognition
          </h1>
          <p className="text-gray-600">
            Draw a digit (0-9) on the canvas and let our models analyze it
          </p>
        </div>
        
        <div className="bg-white shadow-md rounded-lg p-6 mb-8">
          <h2 className="text-xl font-bold text-black mb-4">
            Select Recognition Model
          </h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {models.map(model => (
              <button
                key={model.id}
                className={`p-4 rounded-lg font-bold transition-all ${
                  selectedModel === model.id 
                    ? 'bg-[#5C5C99] text-white shadow-lg' 
                    : 'bg-[#CCCCFF] text-black hover:bg-[#AAAADD]'
                }`}
                onClick={() => setSelectedModel(model.id)}
              >
                {model.name}
              </button>
            ))}
          </div>
        </div>
        
        <DigitRecognitionCanvas 
          modelName={selectedModel} 
          apiEndpoint="/api/recognize-digit"
        />
        
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>Draw a clear digit in the canvas and click "Analyze" to recognize it.</p>
          <p>For best results, draw a centered digit that fills most of the canvas.</p>
        </div>
      </div>
    </div>
  );
};

export default DigitRecognizerPage;