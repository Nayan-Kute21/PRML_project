import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const ImageUpload = ({ setAnalysisResults }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // API base URL
  const API_BASE_URL = 'http://localhost:5000';
  
  // Hardcoded model to SVM
  const modelId = 'SVM';

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setError(null); // Clear any previous errors
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedImage) {
      setUploadStatus('Please select an image first.');
      return;
    }
    
    // Create FormData
    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('model', modelId);
    
    setIsUploading(true);
    setUploadStatus('Analyzing...');
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze image');
      }
      
      const results = await response.json();
      results.imageData = URL.createObjectURL(selectedImage);
      
      setAnalysisResults(results);
      setUploadStatus('Analysis complete!');
      navigate('/results');
    } catch (error) {
      console.error("API Error:", error);
      setError(error.message || 'An unknown error occurred');
      setUploadStatus('Analysis failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="py-8 max-w-5xl mx-auto">
      {/* Main Title */}
      <h1 className="text-4xl font-bold text-[#292966] mb-4 text-center">
        Leaf Classification System
      </h1>
      <p className="text-[#5C5C99] text-lg mb-12 text-center max-w-3xl mx-auto">
        Upload a leaf image to identify its species using our advanced classification system
      </p>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Side - Model Info */}
        <div className="bg-white p-8 rounded-2xl shadow-lg">
          <div className="flex items-center mb-6">
            <div className="bg-[#EEEEFF] p-4 rounded-lg mr-4">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-10 h-10">
                <circle cx="8" cy="8" r="1.5" fill="#CCCCFF" />
                <circle cx="6" cy="12" r="1.5" fill="#CCCCFF" />
                <circle cx="9" cy="16" r="1.5" fill="#CCCCFF" />
                <circle cx="16" cy="7" r="1.5" fill="#5C5C99" />
                <circle cx="18" cy="12" r="1.5" fill="#5C5C99" />
                <circle cx="15" cy="17" r="1.5" fill="#5C5C99" />
                <path d="M4 12C4 12 8 6 20 12" stroke="#292966" strokeWidth="2" fill="none" />
              </svg>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-[#292966]">Support Vector Machine</h2>
              <p className="text-[#5C5C99] font-medium">Classification Model</p>
            </div>
          </div>
          
          <p className="text-gray-700 mb-6 leading-relaxed">
            Support Vector Machine (SVM) is a powerful machine learning algorithm that finds the optimal
            hyperplane to separate different classes in a high-dimensional feature space. It's highly effective
            for leaf classification due to its ability to handle complex patterns.
          </p>
          
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-[#F6F6FF] p-4 rounded-lg text-center">
              <span className="block text-sm text-[#5C5C99] font-medium">Accuracy</span>
              <span className="block text-2xl font-bold text-[#292966]">99%</span>
            </div>
            <div className="bg-[#F6F6FF] p-4 rounded-lg text-center">
              <span className="block text-sm text-[#5C5C99] font-medium">F1 Score</span>
              <span className="block text-2xl font-bold text-[#292966]">0.98</span>
            </div>
          </div>
          
          <ul className="space-y-2 text-[#5C5C99]">
            <li className="flex items-center">
              <svg className="w-5 h-5 mr-2 text-[#292966]" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path></svg>
              High accuracy for species classification
            </li>
            <li className="flex items-center">
              <svg className="w-5 h-5 mr-2 text-[#292966]" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path></svg>
              Handles complex leaf features effectively
            </li>
            <li className="flex items-center">
              <svg className="w-5 h-5 mr-2 text-[#292966]" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path></svg>
              Fast classification response time
            </li>
          </ul>
        </div>

        {/* Right Side - Upload Form */}
        <div className="bg-gradient-to-br from-[#5C5C99] to-[#292966] p-8 rounded-2xl shadow-lg text-white">
          <h2 className="text-2xl font-bold mb-6">Upload Leaf Image</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label 
                htmlFor="imageInput" 
                className="block mb-2 font-medium"
              >
                Select an Image to Classify
              </label>
              <div className="rounded-lg border-2 border-dashed border-[#8585AD] p-6 text-center cursor-pointer hover:border-white transition-colors">
                <input
                  type="file"
                  id="imageInput"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="hidden"
                />
                <label htmlFor="imageInput" className="cursor-pointer">
                  <svg className="mx-auto h-12 w-12 text-[#CCCCFF]" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <p className="mt-2 text-sm">
                    {selectedImage ? selectedImage.name : "Click to upload or drag and drop"}
                  </p>
                  <p className="mt-1 text-xs text-[#CCCCFF]">
                    PNG, JPG, JPEG up to 10MB
                  </p>
                </label>
              </div>
            </div>
            
            {/* Preview Section */}
            {selectedImage && (
              <div className="mb-6">
                <h3 className="block mb-2 font-medium">
                  Image Preview
                </h3>
                <div className="bg-white/10 backdrop-blur-sm p-4 rounded-lg border border-white/20 flex items-center justify-center">
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="Preview"
                    className="max-w-full h-auto max-h-48 rounded"
                  />
                </div>
              </div>
            )}
            
            <button
              type="submit"
              disabled={isUploading || !selectedImage}
              className={`w-full py-3 px-4 rounded-lg font-medium text-center flex items-center justify-center cursor-pointer
                ${(!selectedImage || isUploading) 
                  ? 'bg-[#A3A3CC] cursor-not-allowed opacity-70 text-white/70' 
                  : 'bg-white text-[#292966] hover:bg-[#CCCCFF] transition-colors'}`}
            >
              {isUploading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-[#292966]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </>
              ) : 'Analyze Leaf Image'}
            </button>
            
            {uploadStatus && (
              <div className={`p-4 rounded-lg text-center font-medium
                ${error ? 'bg-red-100 text-red-800' : 
                  uploadStatus.includes('complete') ? 'bg-green-100 text-green-800' : 
                  'bg-blue-100/20 text-white'}`}>
                {error || uploadStatus}
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;