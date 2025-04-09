import { useState, useEffect, useRef } from 'react';

const SignatureCanvas = () => {
  const canvasRef = useRef(null);
  const [ctx, setCtx] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const [textColor, setTextColor] = useState('#3498db');
  const [bgColor, setBgColor] = useState('#ffffff');
  const [lineWidth, setLineWidth] = useState(5);

  // Initialize canvas when component mounts
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    setCtx(context);

    const handleResize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = 400;
      
      // Save current state before resizing
      const currentDrawing = localStorage.getItem('currentSignature');
      
      // Set background after resize
      context.fillStyle = bgColor;
      context.fillRect(0, 0, canvas.width, canvas.height);
      
      // Restore previous drawing
      if (currentDrawing) {
        const img = new Image();
        img.src = currentDrawing;
        img.onload = () => {
          context.drawImage(img, 0, 0);
        };
      }
      
      // Restore stroke settings
      context.strokeStyle = textColor;
      context.lineWidth = lineWidth;
      context.lineCap = 'round';
      context.lineJoin = 'round';
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [bgColor, textColor, lineWidth]);

  // Update context settings when color or line width changes
  useEffect(() => {
    if (ctx) {
      ctx.strokeStyle = textColor;
      ctx.lineWidth = lineWidth;
    }
  }, [ctx, textColor, lineWidth]);

  const saveCurrentState = () => {
    if (canvasRef.current) {
      const currentState = canvasRef.current.toDataURL();
      localStorage.setItem('currentSignature', currentState);
      localStorage.setItem('lastSignature', currentState);
    }
  };

  const handleMouseDown = (e) => {
    setIsDrawing(true);
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
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

  const handleMouseUp = () => {
    setIsDrawing(false);
    saveCurrentState();
  };

  const handleMouseOut = () => {
    setIsDrawing(false);
  };

  const handleTextColorChange = (e) => {
    setTextColor(e.target.value);
  };

  const handleBgColorChange = (e) => {
    const newColor = e.target.value;
    setBgColor(newColor);
    
    if (ctx) {
      ctx.fillStyle = newColor;
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      saveCurrentState();
    }
  };

  const handleLineWidthChange = (e) => {
    setLineWidth(parseInt(e.target.value));
  };

  const handleClear = () => {
    if (ctx) {
      saveCurrentState(); // Save before clearing
      ctx.fillStyle = bgColor;
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const handleSave = () => {
    saveCurrentState();
    const link = document.createElement('a');
    link.download = 'signature.png';
    link.href = canvasRef.current.toDataURL();
    link.click();
  };

  const handleRetrieve = () => {
    if (ctx) {
      const lastState = localStorage.getItem('lastSignature');
      if (lastState) {
        const img = new Image();
        img.src = lastState;
        img.onload = () => {
          ctx.fillStyle = bgColor;
          ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          ctx.drawImage(img, 0, 0);
          saveCurrentState();
        };
      }
    }
  };

  const preventContextMenu = (e) => {
    e.preventDefault();
  };

  return (
    <div className="bg-gradient-to-br from-purple-700 to-blue-400 min-h-screen flex items-center justify-center p-6">
      <div className="container mx-auto p-8 max-w-3xl min-w-md bg-white/85 backdrop-blur-xl rounded-3xl shadow-2xl transform perspective-1000 -rotate-x-2 transition-all duration-300">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white/70 rounded-2xl p-4 transition-all duration-300 hover:bg-white/90 hover:-translate-y-1">
            <label className="block mb-2 text-sm font-bold text-gray-700">
              <i className="ri-palette-line mr-2 text-blue-500"></i>Text Color
            </label>
            <input 
              className="w-full border-2 border-blue-500 rounded-xl p-2 bg-white/80 transition-all duration-300 focus:outline-none focus:border-green-500 focus:ring focus:ring-green-500/20" 
              type="color" 
              value={textColor}
              onChange={handleTextColorChange}
            />
          </div>
          
          <div className="bg-white/70 rounded-2xl p-4 transition-all duration-300 hover:bg-white/90 hover:-translate-y-1">
            <label className="block mb-2 text-sm font-bold text-gray-700">
              <i className="ri-checkbox-blank-line mr-2 text-green-500"></i>Background Color
            </label>
            <input 
              className="w-full border-2 border-blue-500 rounded-xl p-2 bg-white/80 transition-all duration-300 focus:outline-none focus:border-green-500 focus:ring focus:ring-green-500/20" 
              type="color" 
              value={bgColor}
              onChange={handleBgColorChange}
            />
          </div>
          
          <div className="bg-white/70 rounded-2xl p-4 transition-all duration-300 hover:bg-white/90 hover:-translate-y-1">
            <label className="block mb-2 text-sm font-bold text-gray-700">
              <i className="ri-pen-nib-line mr-2 text-red-500"></i>Line Width
            </label>
            <select 
              className="w-full border-2 border-blue-500 rounded-xl p-2 bg-white/80 transition-all duration-300 focus:outline-none focus:border-green-500 focus:ring focus:ring-green-500/20"
              value={lineWidth}
              onChange={handleLineWidthChange}
            >
              <option value="1">Thin (1px)</option>
              <option value="2">Light (2px)</option>
              <option value="3">Medium (3px)</option>
              <option value="4">Bold (4px)</option>
              <option value="5">Extra Bold (5px)</option>
              <option value="10">Massive (10px)</option>
            </select>
          </div>
        </div>
        
        <canvas 
          ref={canvasRef}
          className="w-full h-96 bg-gradient-to-br from-white to-gray-100 border-4 border-blue-500 rounded-2xl shadow-lg cursor-crosshair transition-all duration-300 mb-8"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseOut={handleMouseOut}
          onContextMenu={preventContextMenu}
        />
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <button 
            className="flex items-center justify-center gap-2 px-6 py-3 bg-red-500 hover:bg-red-600 rounded-xl shadow-lg text-white font-bold uppercase tracking-wide transition-all duration-300 hover:scale-105 hover:-translate-y-1 hover:shadow-xl"
            onClick={handleClear}
          >
            <i className="ri-delete-bin-line"></i> Clear
          </button>
          
          <button 
            className="flex items-center justify-center gap-2 px-6 py-3 bg-green-500 hover:bg-green-600 rounded-xl shadow-lg text-white font-bold uppercase tracking-wide transition-all duration-300 hover:scale-105 hover:-translate-y-1 hover:shadow-xl"
            onClick={handleSave}
          >
            <i className="ri-download-2-line"></i> Save
          </button>
          
          <button 
            className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-xl shadow-lg text-white font-bold uppercase tracking-wide transition-all duration-300 hover:scale-105 hover:-translate-y-1 hover:shadow-xl"
            onClick={handleRetrieve}
          >
            <i className="ri-history-line"></i> Retrieve
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignatureCanvas;