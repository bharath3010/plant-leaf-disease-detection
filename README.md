# plant-leaf-disease-detection
"Plant Leaf Disease Detection System using AI Algorithms."

## Overview
This project uses AI algorithms to detect diseases in plant leaves. The model is trained using a dataset of leaf images, and it can predict whether a leaf is healthy or diseased.

## Prerequisites
- Python 3.x
- TensorFlow or PyTorch
- NumPy
- OpenCV

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bharath3010/plant-leaf-disease-detection.git
   cd plant-leaf-disease-detection
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Dataset
1. Download From Kaggle
 ```bash
  https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download from Terminal 
  ```bash
   pip install kaggle
   kaggle datasets download -d emmarex/plantdisease
   unzip plantdisease.zip
  

## Usage
1. Train the Model: To train the model on the dataset of leaf images, run:
   ```bash
   python train_model.py
2. Make Predictions: After training, you can predict diseases using new leaf images:
   ```bash
   python predict.py --image data/sample_leaves/test_image.jpg
