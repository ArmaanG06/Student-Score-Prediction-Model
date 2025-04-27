# Student Exam Score Prediction Project

## Project Overview

This project predicts student exam scores based on their lifestyle, study habits, and background factors. It uses supervised machine learning models to estimate performance given inputs such as study hours, sleep patterns, diet quality, parental education, and other behavioral factors.

Two machine learning models were trained and evaluated:

- **Linear Regression**
- **Random Forest Regressor**

The goal was to build a predictive system that takes user inputs and estimates their likely exam performance.

---

## Dataset Description

The dataset contains various student attributes, including:

- Age
- Gender
- Study hours per day
- Social media usage
- Netflix usage
- Part-time job status
- Attendance percentage
- Sleep hours
- Diet quality (Fair, Good, Poor)
- Exercise frequency
- Parental education level (Bachelor, High School, Master)
- Internet quality (Poor, Average, Good)
- Mental health rating
- Extracurricular participation

The target variable is the student's **exam score** (continuous numerical value).

---

## Project Structure

| File | Description |
|:-----|:------------|
| `Student_Score_Prediction_Notebook.ipynb` | Jupyter Notebook for data exploration, preprocessing, model training, and evaluation |
| `predict_exam_score.py` | Standalone script that allows users to input their details and receive exam score predictions |
| `student_habits_performance.csv` | Original dataset |
| `linear_regression_model.pkl` | Saved Linear Regression model |
| `random_forest_regressor_model.pkl` | Saved Random Forest Regressor model |
| `scaler.pkl` | Saved StandardScaler used for feature scaling |

---

## How to Use

1. Clone this repository to your local machine.
2. Make sure Python is installed (recommended version: 3.9+).
3. Install required packages:
   ```bash
   pip install pandas scikit-learn joblib
4. Run the script:
   python predict_exam_score.py
5. Answer the prompted questions

# Model evaluation
Model	                  MAE  	RMSE	R² Score
Linear Regression	      4.18	5.15	0.90
Random Forest Regressor	4.99	6.26	0.85

Linear Regression slightly outperformed the Random Forest Regressor on this datase

