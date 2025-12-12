# CIFAR-10 Image Classifier (PyTorch)

A simple deep learning project where I train a CNN to classify images from the CIFAR-10 dataset.  
Built as part of my AI engineering learning journey.

---

## Features
- Loads CIFAR-10 dataset using torchvision
- Convolutional Neural Network (CNN)
- Tracks training/validation accuracy
- Saves model checkpoints
- Includes a clean training script + utils

---

## Model Architecture
- Convolution layers
- BatchNorm + ReLU
- MaxPooling
- Linear classifier head

(Architecture defined in `src/model.py`.)

---

## Project Structure
src/
model.py # CNN model
train.py # training loop
utils.py # helper functions
notebooks/
exploration.ipynb # dataset preview + experimentation
results/
loss_curve.png # (generated after training)


---

## How to Run

### 1. Install dependencies

### 2. Train the model


---

## Future Improvements
- Add data augmentation  
- Try a deeper CNN or ResNet  
- Deploy as a small web app using Streamlit  

