# American Sign Language (ASL) Detection System

This project implements a deep learning-based system for detecting American Sign Language (ASL) signs from images. The system recognizes all **29 ASL signs** (26 letters A-Z, plus SPACE, DELETE, and NOTHING).

## 🚀 Features

- Supports **GPU acceleration (CUDA)** for faster model training and inference
- Uses **ResNet18** as the base model with pretrained ImageNet weights
- Provides a **GUI** interface for predicting ASL signs from images
- High accuracy on the ASL dataset
- Easy-to-run training and prediction scripts

---

## ✅ Requirements

- Python 3.7+
- CUDA-capable GPU (Optional, for better performance)
- PyTorch with CUDA support (if using GPU)

The dependencies are listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The system uses the ASL Alphabet dataset, which should be placed in the following directory structure:
```
<your_dataset_path>/asl_alphabet_train/asl_alphabet_train/

```

## Usage

### Training the Model

To train the model, run:
```bash
python train.py
```

The training script will:
- Load and preprocess the training data
- Train a ResNet18-based model on GPU
- Save the best model to 'best_model.pth'

### Making Predictions

To make predictions on new images, run:
```bash
python predict.py
```

When prompted, enter the path to the image you want to classify. The system will output the predicted ASL sign.

## 🧠 Model Architecture
Base model: ResNet18 pretrained on ImageNet

Modified output layer: Outputs 29 classes (for 26 letters + SPACE, DELETE, NOTHING)

Loss Function: CrossEntropyLoss

Optimizer: Adam

Learning Rate: 0.001

Batch Size: 32

Training Device: GPU (if available)

## 📈 Performance
The model achieves high accuracy on the ASL dataset. Training progress is displayed in real-time, including:

Current epoch

Training loss

Training accuracy

Progress bar (via tqdm)

The best model based on validation accuracy is automatically saved.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

