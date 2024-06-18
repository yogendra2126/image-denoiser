# Image Denoiser Project

## Overview
This project implements an image denoising model using a convolutional autoencoder. The model is designed to improve the quality of images captured in low-light conditions by reducing noise.

## Project Structure
- **train.zip**: Contains the training dataset with two folders, `low` (noisy images) and `high` (clean images).
- **image_denoiser.ipynb**: Jupyter Notebook with the complete implementation of the image denoiser model, including data loading, preprocessing, model training, and evaluation.
- **report.pdf**: Detailed report on the project, including the architecture, code snippets, training process, evaluation, and results.

## Model Architecture
The convolutional autoencoder architecture used in this project consists of:
- **Encoder**:
  - Conv2D (64 filters, 3x3 kernel, ReLU, BatchNorm)
  - MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3 kernel, ReLU, BatchNorm)
  - MaxPooling2D (2x2)
- **Decoder**:
  - Conv2D (64 filters, 3x3 kernel, ReLU, BatchNorm)
  - UpSampling2D (2x2)
  - Conv2D (64 filters, 3x3 kernel, ReLU, BatchNorm)
  - UpSampling2D (2x2)
  - Concatenate with encoder output
  - Conv2D (3 filters, 3x3 kernel, Sigmoid)

## Results
The model achieved an average Peak Signal-to-Noise Ratio (PSNR) of **18.33** on the test dataset.

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/yogendra2126/image-denoiser.git
    ```
2. Extract the `train.zip` file into the repository directory.
3. Open and run the `image_denoiser.ipynb` notebook to train the model and evaluate the results.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- PIL
- Jupyter

Install the required packages using:
```sh
pip install tensorflow numpy matplotlib pillow jupyter
