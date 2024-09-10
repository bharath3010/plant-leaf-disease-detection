
# Plant Leaf Disease Detection System

## Overview
This project implements AI algorithms to detect diseases in plant leaves. By analyzing a dataset of leaf images, the system predicts whether a leaf is healthy or affected by diseases. The primary goal is to aid early diagnosis, improving agricultural productivity by enabling timely intervention.

## Prerequisites
- Python 3.x
- TensorFlow or PyTorch (choose based on your framework)
- NumPy
- OpenCV
- Scikit-learn (optional for preprocessing)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bharath3010/plant-leaf-disease-detection.git
   cd plant-leaf-disease-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
1. Download the dataset from Kaggle:
   [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

2. **Alternatively**, download the dataset directly using the terminal:

   - Install Kaggle CLI:
     ```bash
     pip install kaggle
     ```

   - Download the dataset:
     ```bash
     kaggle datasets download -d emmarex/plantdisease
     ```

   - Unzip the dataset:
     ```bash
     unzip plantdisease.zip
     ```

## Usage

### 1. Train the Model
To train the model on the plant leaf dataset, run the following command:

```bash
python train_model.py
```

### 2. Make Predictions
After training the model, you can make predictions on new leaf images to detect diseases:

```bash
python predict.py --image data/sample_leaves/test_image.jpg
```

This will output whether the leaf is healthy or diseased, along with the type of disease (if applicable).

## Model Architecture
The model used in this project is a Convolutional Neural Network (CNN), which is suitable for image classification tasks. It consists of several convolutional layers followed by pooling and dense layers for classification.

## Evaluation
The model’s accuracy and performance are evaluated using standard metrics such as accuracy, precision, recall, and F1 score. A validation split is also used during training to monitor overfitting.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License.
