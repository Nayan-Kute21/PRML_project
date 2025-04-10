# Leaf Classification Project

This project aims to classify leaf images into different species using various machine learning techniques.

## Project Overview

This repository contains an implementation of multiple machine learning models for leaf species classification based on the Kaggle dataset: https://www.kaggle.com/competitions/leaf-classification

## Directory Structure

```
leaf-classification/
│
├── data/                      # Data directory
│   ├── raw/                   # Original dataset files
│   └── processed/             # Processed data files
│
├── models/                    # Trained model storage
│   ├── knn/
│   ├── svm/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── ann/
│   └── cnn/
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training_evaluation.ipynb
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data preprocessing functions
│   ├── feature_extraction.py  # Feature extraction utilities
│   ├── model_training.py      # Model training functions
│   ├── evaluation.py          # Model evaluation metrics
│   └── visualization.py       # Visualization utilities
│
├── results/                   # Results and comparison analysis
│   ├── metrics/               # Performance metrics
│   ├── visualizations/        # Visualization outputs
│   └── failure_analysis/      # Failure case analysis
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. Clone this repository:
```
git clone https://github.com/yourusername/leaf-classification.git
cd leaf-classification
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Download the dataset from Kaggle: https://www.kaggle.com/competitions/leaf-classification
   - Place the files in the `data/raw/` directory
   - Make sure you have train.csv, test.csv, and the images folder

5. Run the notebooks in the `notebooks/` directory or execute the main script:
```
python src/main.py
```

## Models Implemented

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. Artificial Neural Network (ANN)
6. Convolutional Neural Network (CNN)

## Results

The model performance comparison and analysis can be found in the `results/` directory after running the code.