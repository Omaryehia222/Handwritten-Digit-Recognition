# Handwritten Digit Recognition

A deep learning application that recognizes handwritten digits drawn by the user through a simple desktop GUI.

## Overview

The application loads a pre-trained CNN model and provides a Tkinter drawing canvas. The user draws a digit, clicks **Predict**, and the application preprocesses the drawing into the MNIST input format before predicting the digit.

## Features

- Interactive desktop drawing canvas
- Handwritten digit prediction from 0 to 9
- Image resizing to `28×28`
- Grayscale image preprocessing and color inversion
- CNN model inference using TensorFlow/Keras
- Clear button to reset the canvas

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Tkinter
- MNIST-compatible `28×28×1` input

## Model

The included `mnist_cnn_model.h5` is a trained Sequential CNN model. Its architecture contains:

- Conv2D: 32 filters, `3×3` kernel, ReLU
- MaxPooling2D
- Flatten
- Dense: 128 units, ReLU
- Dense: 10 units, Softmax

## Project Structure

```text
Handwritten-Digit-Recognition/
├── digit_gui.py
├── mnist_cnn_model.h5
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Install Python 3.x.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure `digit_gui.py` and `mnist_cnn_model.h5` are in the same folder.

## Run

```bash
python digit_gui.py
```

Draw a digit on the canvas and click **Predict**.

## Notes

The application expects the saved model to be named `mnist_cnn_model.h5` and located in the same directory as `digit_gui.py`.
