import { useState } from 'react';

export default function ImageUploadComponent({ modelNumber = '' }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  

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
    formData.append('modelNumber', modelNumber); // Using the model number passed as prop
    
    setIsUploading(true);
    setUploadStatus('Uploading...');
    
    try {
      // Mock API endpoint
      const response = await fetch('https://api.example.com/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        setUploadStatus('Upload successful!');
        // Reset form after successful upload
        setSelectedImage(null);
        // Reset the file input
        document.getElementById('imageInput').value = '';
      } else {
        setUploadStatus('Upload failed. Please try again.');
      }
    } catch (error) {
      setUploadStatus('Error uploading: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    
    <div className="flex flex-col items-center justify-center p-8 rounded-lg shadow-lg" style={{ backgroundColor: '#5c5c99' }}>
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'black' }}>Image Upload</h2>
      
      <form onSubmit={handleSubmit} className="w-full max-w-md">
        <div className="mb-4">
          <label htmlFor="imageInput" className="block mb-2 font-medium" style={{ color: 'black' }}>
            Select Image
          </label>
          <input
            type="file"
            id="imageInput"
            accept="image/*"
            onChange={handleImageChange}
            className="w-full p-2 border rounded bg-white"
            style={{ color: 'black' }}
          />
        </div>
        
        {/* Preview Section - Always visible */}
        <div className="mb-4">
          <h3 className="block mb-2 font-medium" style={{ color: 'black' }}>
            Image Preview
          </h3>
          <div 
            className="bg-white p-4 rounded border-2 border-dashed flex items-center justify-center"
            style={{ minHeight: '200px' }}
          >
            {selectedImage ? (
              <img
                src={URL.createObjectURL(selectedImage)}
                alt="Preview"
                className="max-w-full h-auto max-h-48"
              />
            ) : (
              <p className="text-gray-500">No image selected</p>
            )}
          </div>
        </div>
        
        <button
          type="submit"
          disabled={isUploading}
          className="w-full py-2 px-4 rounded font-medium mt-4"
          style={{ backgroundColor: '#ccccff', color: 'black' }}
        >
          {isUploading ? 'Uploading...' : 'Upload Image'}
        </button>
        
        {uploadStatus && (
          <p className="mt-4 text-center font-medium" style={{ color: 'black' }}>
            {uploadStatus}
          </p>
        )}
      </form>
    </div>
  );
}