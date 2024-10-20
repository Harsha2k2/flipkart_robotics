# Fruit Freshness Prediction API

## Overview

The **Fruit Freshness Prediction API** is a Flask-based web application that utilizes a trained PyTorch model to determine the freshness of fruits based on uploaded images. The API classifies fruits as either "Fresh" or "Rotten," making it a valuable tool for food quality assessment and inventory management.

## Features

- **Image Upload**: Users can upload images of fruits to check their freshness.
- **Real-time Predictions**: The API provides immediate feedback on the freshness status of the uploaded fruit.
- **Built with Flask**: A lightweight and easy-to-use web framework for Python.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)



Expiry date detection can be very helpful in many real-world scenarios. Either to aid the visually impaired or to just make it easier to read expiry dates which could be printed in various formats and fonts, implementing an automated date detector could be beneficial. This work utilized concepts of computer vision and regression models in order to detect the category of the object and then the expiry date printed on it. Initially, PCA was used to compress the images as part of unsupervised learning. A fine-tuned ResNet-50 model pretrained on the ImageNet dataset was used for the supervised object classification task while the VGG-16 regression model was used in order to produce bounding boxes to detect the region of the expiry date which was then generated as text output using OCR. Although the ResNet-50 model overfitted the data initially, adding dropout layers, tuning the hyperparameters and increasing the training data size using image augmentation helped in overcoming overfitting. This work could be extended to using more recent models such as AlexNet and InceptionNet for classification to see if there will be any performance improvement.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/fruit-freshness-prediction.git
   cd fruit-freshness-prediction
