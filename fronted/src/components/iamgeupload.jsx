import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const ImageUpload = ({ modelId, setAnalysisResults }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const navigate = useNavigate();

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
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
    formData.append('modelId', modelId);
    
    setIsUploading(true);
    setUploadStatus('Analyzing...');
    
    try {
      // Mock API endpoint - replace with your actual API
      // For demo purposes, we'll simulate a response after a delay
      setTimeout(() => {
        // Mock response data
        const mockResults = {
          modelId: modelId,
          predictions: [
            { class: 'Oak', probability: 0.89 },
            { class: 'Maple', probability: 0.08 },
            { class: 'Birch', probability: 0.03 },
          ],
          processingTime: '1.2s',
          imageData: URL.createObjectURL(selectedImage)
        };
        
        setAnalysisResults(mockResults);
        setUploadStatus('Analysis complete!');
        navigate('/results'); // Navigate to results page
      }, 2000);
      
      // Uncomment below for actual API implementation
      /*
      const response = await fetch('https://api.example.com/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const results = await response.json();
        setAnalysisResults(results);
        setUploadStatus('Analysis complete!');
        navigate('/results');
      } else {
        setUploadStatus('Analysis failed. Please try again.');
      }
      */
    } catch (error) {
      setUploadStatus('Error: ' + error.message);
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
              ${uploadStatus.includes('Error') ? 'bg-red-100 text-red-800' : 
                uploadStatus.includes('complete') ? 'bg-green-100 text-green-800' : 
                'bg-blue-100 text-blue-800'}`}>
              {uploadStatus}
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default ImageUpload;