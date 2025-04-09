import { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

const DigitRecognitionCanvas = ({ modelName, apiEndpoint = '/api/recognize-digit' }) => {
  const canvasRef = useRef(null);
  const [ctx, setCtx] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [lineWidth, setLineWidth] = useState(5);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Initialize canvas when component mounts
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    setCtx(context);

    const handleResize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = 400;
      
      // Set background after resize
      context.fillStyle = '#FFFFFF';
      context.fillRect(0, 0, canvas.width, canvas.height);
      
      // Restore previous drawing
      const currentDrawing = localStorage.getItem('currentDigit');
      if (currentDrawing) {
        const img = new Image();
        img.src = currentDrawing;
        img.onload = () => {
          context.drawImage(img, 0, 0);
        };
      }
      
      // Configure for grayscale drawing
      context.strokeStyle = '#000000';
      context.lineWidth = lineWidth;
      context.lineCap = 'round';
      context.lineJoin = 'round';
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [lineWidth]);

  // Update line width when it changes
  useEffect(() => {
    if (ctx) {
      ctx.lineWidth = lineWidth;
    }
  }, [ctx, lineWidth]);

  const saveCurrentState = () => {
    if (canvasRef.current) {
      const currentState = canvasRef.current.toDataURL();
      localStorage.setItem('currentDigit', currentState);
    }
  };

  const handleMouseDown = (e) => {
    setIsDrawing(true);
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setLastPos({ x, y });
  };

  const handleTouchStart = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const y = e.touches[0].clientY - rect.top;
    setLastPos({ x, y });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !ctx) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastPos.x, lastPos.y);
    ctx.lineTo(x, y);
    ctx.stroke();

    setLastPos({ x, y });
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    if (!isDrawing || !ctx) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const y = e.touches[0].clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastPos.x, lastPos.y);
    ctx.lineTo(x, y);
    ctx.stroke();

    setLastPos({ x, y });
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    saveCurrentState();
  };

  const handleTouchEnd = () => {
    setIsDrawing(false);
    saveCurrentState();
  };

  const handleMouseOut = () => {
    setIsDrawing(false);
  };

  const handleLineWidthChange = (e) => {
    setLineWidth(parseInt(e.target.value));
  };

  const handleClear = () => {
    if (ctx) {
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      saveCurrentState();
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    try {
      setIsAnalyzing(true);
      setError(null);
      
      // Get the canvas data as a blob
      const imageBlob = await new Promise(resolve => {
        canvasRef.current.toBlob(resolve, 'image/png');
      });
      
      // Create FormData and append the image and model name
      const formData = new FormData();
      formData.append('image', imageBlob, 'digit.png');
      formData.append('model', modelName);
      
      // Send to backend
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Error analyzing digit:', err);
      setError(err.message || 'Failed to analyze digit');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const preventContextMenu = (e) => {
    e.preventDefault();
  };

  return (
    <div className="bg-[#5C5C99] p-6 rounded-2xl shadow-xl">
      <h2 className="font-arial font-bold text-2xl text-black mb-4 text-center">
        Digit Recognition
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white/70 rounded-xl p-4">
          <label className="block mb-2 text-sm font-bold text-black font-arial">
            Line Width
          </label>
          <select 
            className="w-full border-2 border-[#5C5C99] rounded-xl p-2 bg-white font-arial font-bold text-black"
            value={lineWidth}
            onChange={handleLineWidthChange}
          >
            <option value="1">Thin (1px)</option>
            <option value="3">Light (3px)</option>
            <option value="5">Medium (5px)</option>
            <option value="8">Bold (8px)</option>
            <option value="12">Extra Bold (12px)</option>
            <option value="18">Massive (18px)</option>
          </select>
        </div>
        
        {result && (
          <div className="bg-white/70 rounded-xl p-4">
            <h3 className="font-arial font-bold text-black mb-2">Result:</h3>
            <div className="bg-[#CCCCFF] p-3 rounded-lg">
              <p className="font-arial font-bold text-2xl text-center">
                Recognized digit: <span className="text-3xl">{result.digit}</span>
              </p>
              {result.confidence && (
                <p className="font-arial text-sm text-center mt-1">
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </p>
              )}
            </div>
          </div>
        )}
      </div>
      
      <div className="relative mb-6">
        <canvas 
          ref={canvasRef}
          className="w-full h-96 bg-white border-4 border-[#CCCCFF] rounded-xl shadow-lg cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseOut={handleMouseOut}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onContextMenu={preventContextMenu}
        />
        
        {isAnalyzing && (
          <div className="absolute inset-0 bg-black/30 flex items-center justify-center rounded-xl">
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <p className="font-arial font-bold text-black">Analyzing...</p>
            </div>
          </div>
        )}
      </div>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6 font-arial">
          <p className="font-bold">Error:</p>
          <p>{error}</p>
        </div>
      )}
      
      <div className="grid grid-cols-2 gap-6">
        <button 
          className="flex items-center justify-center gap-2 px-6 py-3 bg-[#CCCCFF] hover:bg-[#AAAADD] rounded-xl shadow-lg text-black font-arial font-bold tracking-wide transition-all duration-300 hover:scale-105 hover:shadow-xl"
          onClick={handleClear}
        >
          Clear
        </button>
        
        <button 
          className="flex items-center justify-center gap-2 px-6 py-3 bg-[#CCCCFF] hover:bg-[#AAAADD] rounded-xl shadow-lg text-black font-arial font-bold tracking-wide transition-all duration-300 hover:scale-105 hover:shadow-xl"
          onClick={handleAnalyze}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
    </div>
  );
};

DigitRecognitionCanvas.propTypes = {
  modelName: PropTypes.string.isRequired,
  apiEndpoint: PropTypes.string
};

export default DigitRecognitionCanvas;