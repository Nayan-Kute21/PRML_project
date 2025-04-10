import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const ImageUpload = ({ modelId, setAnalysisResults }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // API base URL - adjust based on your backend location
  const API_BASE_URL = 'http://localhost:5000';

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
    formData.append('model', modelId); // Use modelId as the model parameter
    
    setIsUploading(true);
    setUploadStatus('Analyzing...');
    setError(null);
    console.log("modelId", modelId);  
    try {
      // Make actual API call to the backend
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        // No need to set Content-Type header as FormData will set it automatically
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze image');
      }
      
      const results = await response.json();
      
      // Add the image URL to the results
      results.imageData = URL.createObjectURL(selectedImage);
      
      setAnalysisResults(results);
      setUploadStatus('Analysis complete!');
      navigate('/results'); // Navigate to results page
    } catch (error) {
      console.error("API Error:", error);
      setError(error.message || 'An unknown error occurred');
      setUploadStatus('Analysis failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto my-8">
      <div className="flex flex-col items-center justify-center p-8 rounded-lg shadow-lg bg-gradient-to-br from-[#5C5C99] to-[#292966]">
        <h2 className="text-2xl font-bold mb-6 text-white">Upload Leaf Image</h2>
        
        <form onSubmit={handleSubmit} className="w-full">
          <div className="mb-6">
            <label 
              htmlFor="imageInput" 
              className="block mb-2 font-medium text-white"
            >
              Select Image
            </label>
            <div className="relative">
              <input
                type="file"
                id="imageInput"
                accept="image/*"
                onChange={handleImageChange}
                className="w-full p-3 border rounded bg-white text-[#292966] file:mr-4 file:py-2 file:px-4
                file:rounded file:border-0 file:bg-[#CCCCFF] file:text-[#292966] file:font-medium
                hover:file:bg-[#A3A3CC] file:cursor-pointer cursor-pointer"
              />
            </div>
          </div>
          
          {/* Preview Section */}
          <div className="mb-6">
            <h3 className="block mb-2 font-medium text-white">
              Image Preview
            </h3>
            <div 
              className="bg-white p-4 rounded-lg border-2 border-dashed border-[#CCCCFF] flex items-center justify-center"
              style={{ minHeight: '200px' }}
            >
              {selectedImage ? (
                <img
                  src={URL.createObjectURL(selectedImage)}
                  alt="Preview"
                  className="max-w-full h-auto max-h-48 rounded"
                />
              ) : (
                <p className="text-[#5C5C99]">No image selected</p>
              )}
            </div>
          </div>
          
          <button
            type="submit"
            disabled={isUploading || !selectedImage}
            className={`w-full py-3 px-4 rounded-lg font-medium mt-4 text-[#292966] transition-all
              ${(!selectedImage || isUploading) 
                ? 'bg-[#A3A3CC] cursor-not-allowed opacity-70' 
                : 'bg-[#CCCCFF] hover:bg-white'}`}
          >
            {isUploading ? 'Analyzing...' : 'Analyze Image'}
          </button>
          
          {uploadStatus && (
            <div className={`mt-4 p-3 rounded-lg text-center font-medium
              ${error ? 'bg-red-100 text-red-800' : 
                uploadStatus.includes('complete') ? 'bg-green-100 text-green-800' : 
                'bg-blue-100 text-blue-800'}`}>
              {error || uploadStatus}
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default ImageUpload;