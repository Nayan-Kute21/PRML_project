import { useState } from 'react';
import PropTypes from 'prop-types';

// Icons for different models
const modelIcons = {
  KNN: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
      <circle cx="6" cy="8" r="2" fill="#CCCCFF" />
      <circle cx="14" cy="6" r="2" fill="#CCCCFF" />
      <circle cx="18" cy="12" r="2" fill="#CCCCFF" />
      <circle cx="10" cy="16" r="2" fill="#CCCCFF" />
      <circle cx="16" cy="18" r="2" fill="#CCCCFF" />
      <circle cx="10" cy="10" r="2.5" fill="#292966" />
      <line x1="10" y1="10" x2="6" y2="8" stroke="#A3A3CC" strokeWidth="1" />
      <line x1="10" y1="10" x2="14" y2="6" stroke="#A3A3CC" strokeWidth="1" />
      <line x1="10" y1="10" x2="18" y2="12" stroke="#A3A3CC" strokeWidth="1" />
      <line x1="10" y1="10" x2="10" y2="16" stroke="#A3A3CC" strokeWidth="1" />
      <line x1="10" y1="10" x2="16" y2="18" stroke="#A3A3CC" strokeWidth="1" />
    </svg>
  ),
  DT: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
      <circle cx="12" cy="4" r="2" fill="#292966" />
      <line x1="12" y1="6" x2="12" y2="8" stroke="#5C5C99" strokeWidth="1.5" />
      <circle cx="12" cy="10" r="2" fill="#5C5C99" />
      <line x1="12" y1="12" x2="8" y2="15" stroke="#5C5C99" strokeWidth="1.5" />
      <line x1="12" y1="12" x2="16" y2="15" stroke="#5C5C99" strokeWidth="1.5" />
      <circle cx="8" cy="17" r="2" fill="#A3A3CC" />
      <circle cx="16" cy="17" r="2" fill="#A3A3CC" />
      <line x1="8" y1="19" x2="6" y2="22" stroke="#A3A3CC" strokeWidth="1.5" />
      <line x1="8" y1="19" x2="10" y2="22" stroke="#A3A3CC" strokeWidth="1.5" />
      <line x1="16" y1="19" x2="14" y2="22" stroke="#A3A3CC" strokeWidth="1.5" />
      <line x1="16" y1="19" x2="18" y2="22" stroke="#A3A3CC" strokeWidth="1.5" />
    </svg>
  ),
  SVM: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
      <circle cx="8" cy="8" r="1.5" fill="#CCCCFF" />
      <circle cx="6" cy="12" r="1.5" fill="#CCCCFF" />
      <circle cx="9" cy="16" r="1.5" fill="#CCCCFF" />
      <circle cx="16" cy="7" r="1.5" fill="#5C5C99" />
      <circle cx="18" cy="12" r="1.5" fill="#5C5C99" />
      <circle cx="15" cy="17" r="1.5" fill="#5C5C99" />
      <path d="M4 12C4 12 8 6 20 12" stroke="#292966" strokeWidth="2" fill="none" />
    </svg>
  ),
  RF: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
      <path d="M6 20V10L10 6L14 10V20" fill="none" stroke="#CCCCFF" strokeWidth="1.5" />
      <path d="M10 20V8L14 4L18 8V20" fill="none" stroke="#A3A3CC" strokeWidth="1.5" />
      <path d="M2 20V12L6 8L10 12V20" fill="none" stroke="#5C5C99" strokeWidth="1.5" />
      <line x1="2" y1="20" x2="18" y2="20" stroke="#292966" strokeWidth="1.5" />
    </svg>
  ),
  ANN: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
      <circle cx="6" cy="6" r="1.5" fill="#CCCCFF" />
      <circle cx="6" cy="12" r="1.5" fill="#CCCCFF" />
      <circle cx="6" cy="18" r="1.5" fill="#CCCCFF" />
      <circle cx="12" cy="9" r="1.5" fill="#A3A3CC" />
      <circle cx="12" cy="15" r="1.5" fill="#A3A3CC" />
      <circle cx="18" cy="12" r="1.5" fill="#292966" />
      <line x1="6" y1="6" x2="12" y2="9" stroke="#5C5C99" strokeWidth="1" />
      <line x1="6" y1="6" x2="12" y2="15" stroke="#5C5C99" strokeWidth="1" />
      <line x1="6" y1="12" x2="12" y2="9" stroke="#5C5C99" strokeWidth="1" />
      <line x1="6" y1="12" x2="12" y2="15" stroke="#5C5C99" strokeWidth="1" />
      <line x1="6" y1="18" x2="12" y2="9" stroke="#5C5C99" strokeWidth="1" />
      <line x1="6" y1="18" x2="12" y2="15" stroke="#5C5C99" strokeWidth="1" />
      <line x1="12" y1="9" x2="18" y2="12" stroke="#5C5C99" strokeWidth="1" />
      <line x1="12" y1="15" x2="18" y2="12" stroke="#5C5C99" strokeWidth="1" />
    </svg>
  ),
  CNN: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="w-full h-full">
      <rect x="3" y="6" width="4" height="4" fill="#CCCCFF" stroke="#A3A3CC" strokeWidth="1" />
      <rect x="3" y="14" width="4" height="4" fill="#CCCCFF" stroke="#A3A3CC" strokeWidth="1" />
      <rect x="10" y="6" width="4" height="4" fill="#A3A3CC" stroke="#5C5C99" strokeWidth="1" />
      <rect x="10" y="14" width="4" height="4" fill="#A3A3CC" stroke="#5C5C99" strokeWidth="1" />
      <rect x="17" y="10" width="4" height="4" fill="#292966" stroke="#5C5C99" strokeWidth="1" />
      <line x1="7" y1="8" x2="10" y2="8" stroke="#5C5C99" strokeWidth="1" />
      <line x1="7" y1="16" x2="10" y2="16" stroke="#5C5C99" strokeWidth="1" />
      <line x1="14" y1="8" x2="17" y2="10" stroke="#5C5C99" strokeWidth="1" />
      <line x1="14" y1="16" x2="17" y2="14" stroke="#5C5C99" strokeWidth="1" />
    </svg>
  ),
};

const ModelButton = ({ 
  model, 
  name, 
  description, 
  stats, 
  isSelected, 
  onClick 
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  // Base colors from the provided color scheme
  const colors = {
    lightest: '#CCCCFF',
    light: '#A3A3CC',
    medium: '#5C5C99',
    dark: '#292966'
  };

  // Dynamic styles based on selection state
  const getBackgroundColor = () => {
    if (isSelected) return colors.medium;
    if (isHovered) return colors.light;
    return colors.lightest;
  };

  const getTextColor = () => {
    if (isSelected) return 'white';
    return colors.dark;
  };

  // Format stat values as percentages if they're between 0-1
  const formatStatValue = (value, index) => {
    // First stat should be displayed as percentage
    if (index === 0 && typeof value === 'number') {
      return `${(value * 100).toFixed(1)}%`;
    }
    // Second stat should be displayed as is
    return value;
  };

  return (
    <div 
      className={`
        rounded-lg overflow-hidden shadow-lg transition-all duration-200 cursor-pointer
        ${isSelected ? 'transform scale-102 shadow-xl ring-2 ring-offset-2 ring-[#5C5C99]' : ''}
      `}
      style={{ 
        backgroundColor: getBackgroundColor(),
        transform: isSelected ? 'scale(1.02)' : (isHovered ? 'scale(1.01)' : 'scale(1)'),
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
    >
      <div className="p-4">
        {/* Top section with icon and name */}
        <div className="flex items-start mb-3">
          {/* Model Icon */}
          <div className="w-16 h-16 p-2 bg-white rounded-lg shadow-sm flex items-center justify-center">
            {modelIcons[model] || (
              <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center">
                <span className="text-[#5C5C99] text-xs font-bold">{model}</span>
              </div>
            )}
          </div>
          
          {/* Model Name */}
          <div 
            className="ml-3 flex-grow rounded-lg p-3 flex items-center"
            style={{ 
              backgroundColor: isSelected ? colors.dark : 'rgba(255, 255, 255, 0.7)',
              color: isSelected ? 'white' : colors.dark
            }}
          >
            <h3 className="text-lg font-bold">{name}</h3>
          </div>
        </div>

        {/* Description */}
        <div className="mb-3">
          <p className="text-sm" style={{ color: getTextColor() }}>{description}</p>
        </div>

        {/* Stats */}
        <div 
          className="rounded-lg p-3 space-y-2"
          style={{ backgroundColor: 'rgba(255, 255, 255, 0.3)' }}
        >
          {stats && Object.entries(stats).map(([key, value]) => (
            <div key={key} className="flex justify-between items-center">
              <span className="text-xs font-semibold" style={{ color: colors.dark }}>
                {key}:
              </span>
              <span 
                className="text-xs font-bold px-2 py-1 rounded"
                style={{ 
                  backgroundColor: isSelected ? colors.dark : colors.medium,
                  color: 'white'
                }}
              >
                {formatStatValue(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

ModelButton.propTypes = {
  model: PropTypes.oneOf(['knn', 'decisionTree', 'svm', 'randomForest', 'ann', 'cnn']).isRequired,
  name: PropTypes.string.isRequired,
  description: PropTypes.string,
  stats: PropTypes.object,
  isSelected: PropTypes.bool,
  onClick: PropTypes.func
};

ModelButton.defaultProps = {
  description: 'No description available',
  stats: {},
  isSelected: false,
  onClick: () => {}
};

export default ModelButton;