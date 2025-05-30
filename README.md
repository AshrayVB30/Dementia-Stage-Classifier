﻿# Dementia-Stage-Classifier

## Overview

The **Dementia Stage Classifier** is a deep learning-based application that predicts the stage of dementia from brain MRI images. The model classifies the dementia stage into four categories:

- **MildDemented**
- **ModerateDemented**
- **NonDemented**
- **VeryMildDemented**

The application allows users to upload an MRI image, and it will predict the dementia stage along with confidence, behavior, and precautions.

## Features

- **MRI Image Upload**: Upload MRI images (JPG, JPEG, PNG formats) to the app.
- **Stage Prediction**: The model predicts the dementia stage with confidence and provides suggestions for behavior and precautions.
- **User-Friendly Interface**: Built using Streamlit for an easy-to-use interface.

## Requirements

The following Python libraries are required to run this project:

- Python 3.9 or higher
- TensorFlow 2.x
- Streamlit
- OpenCV
- Scikit-learn
- Pillow

These dependencies are listed in the `requirements.txt` file.

## Installation

Follow these steps to set up and run the project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AshrayVB30/Dementia-Stage-Classifier.git

## Overview

The **Dementia Stage Classifier** is a deep learning-based application that predicts the stage of dementia from brain MRI images. The model classifies the dementia stage into four categories:

- **MildDemented**
- **ModerateDemented**
- **NonDemented**
- **VeryMildDemented**

The application allows users to upload an MRI image, and it will predict the dementia stage along with confidence, behavior, and precautions.

## Features

- **MRI Image Upload**: Upload MRI images (JPG, JPEG, PNG formats) to the app.
- **Stage Prediction**: The model predicts the dementia stage with confidence and provides suggestions for behavior and precautions.
- **User-Friendly Interface**: Built using Streamlit for an easy-to-use interface.

## Requirements

The following Python libraries are required to run this project:

- Python 3.9 or higher
- TensorFlow 2.x
- Streamlit
- OpenCV
- Scikit-learn
- Pillow

These dependencies are listed in the `requirements.txt` file.

## Installation

Follow these steps to set up and run the project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AshrayVB30/Dementia-Stage-Classifier.git
    cd dementia-stage-classifier
    ```

2. **Create a virtual environment** (optional, but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure


    cd dementia-stage-classifier
    ```

2. **Create a virtual environment** (optional, but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

### 1. **Train the model** (if you don't have the pre-trained model):

   To train the model and save it as `dementia_model.h5`, run the following command:

   ```bash
   python frontend/Training.py
   ```

This will load and process the dataset, then train the model, and finally save it as dementia_model.h5.

2. Start the Streamlit app:
To run the app, use the following command:
```bash
streamlit run frontend/app.py
```

