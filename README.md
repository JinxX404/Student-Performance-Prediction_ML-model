# Student Performance Prediction System

This project predicts student final grades using machine learning models trained on the UCI Student Performance dataset. It includes a Jupyter notebook for data exploration and model training, a Python script for model training, and a Tkinter-based GUI application for making predictions.

## Project Structure

- `model_training.py`: Script for data preprocessing, model training, and evaluation.
- `student_performance_app.py`: Tkinter GUI application for predicting student grades using the trained model.
- `student_performance_model.joblib`: Saved machine learning model.
- `student-mat.csv`: Dataset used for training and evaluation.
- `student_grade_prediction_notebook.ipynb`: Jupyter notebook for exploratory data analysis and model development.

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script (if you want to retrain the model):
   ```bash
   python model_training.py
   ```
4. Launch the GUI application:
   ```bash
   python student_performance_app.py
   ```

## Usage

- The GUI allows you to input student features and predicts the final grade using the trained model.
- The notebook provides step-by-step data analysis and model building.

## Dataset

The dataset (`student-mat.csv`) is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance).
