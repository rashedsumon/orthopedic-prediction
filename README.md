# Orthopedic Patient Condition Classifier 🦴

An interactive Streamlit web application that predicts spine conditions (**Normal**, **Disk Hernia**, or **Spondylolisthesis**) utilizing machine learning model trained on biomechanical features.

## Dataset
This app automatically streams and downloads the [Biomechanical Features of Orthopedic Patients dataset](https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients) from the UCI Machine Learning Repository via `kagglehub` during runtime.

## Local Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt