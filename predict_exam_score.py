import pandas as pd
import numpy as np
import joblib

linear_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_regressor_model.pkl')
scaler = joblib.load('scaler.pkl')

print("Welcome to the Exam Score Predictor!")

student_id = input("Enter your student ID: ")
age = float(input("Enter your age: "))
gender = input("Enter your gender (Male/Female): ")
study_hours_per_day = float(input("Enter your study hours per day: "))
social_media_hours = float(input("Enter your social media hours per day: "))
netflix_hours = float(input("Enter your Netflix hours per day: "))
part_time_job = input("Do you have a part-time job? (Yes/No): ")
attendance_percentage = float(input("Enter your attendance percentage: "))
sleep_hours = float(input("Enter your sleep hours per night: "))
diet_quality = input("How would you describe your diet? (Fair/Good/Poor): ")
exercise_frequency = int(input("How many times you exercise per week?: "))
parental_education_level = input("Parental education level? (High School/Bachelor/Master): ")
internet_quality = input("Internet quality? (Poor/Average/Good): ")
mental_health_rating = int(input("Mental health rating (1-10): "))
extracurricular_participation = input("Extracurricular participation? (Yes/No): ")

#Preping Data
if gender.lower() == 'male':
    gender = 0
else:
    gender = 1

if part_time_job.lower() == "no":
    part_time_job = 0
else:
    part_time_job = 1

if diet_quality.lower() == 'poor':
    diet_quality_Poor = 1
    diet_quality_Fair = 0
    diet_quality_Good = 0

elif diet_quality.lower() == "fair":
    diet_quality_Poor = 0
    diet_quality_Fair = 1
    diet_quality_Good = 0
elif diet_quality.lower() == "good":
    diet_quality_Poor = 0
    diet_quality_Fair = 0
    diet_quality_Good = 1

if parental_education_level.lower() == 'master':
    parental_education_level_High_School = 0
    parental_education_level_Bachelor = 0
    parental_education_level_Master = 1
elif parental_education_level.lower() == 'bachelor':
    parental_education_level_High_School = 0
    parental_education_level_Bachelor = 1
    parental_education_level_Master = 0
elif parental_education_level.lower() == 'high school':
    parental_education_level_High_School = 1
    parental_education_level_Bachelor = 0
    parental_education_level_Master = 0

if internet_quality.lower() == 'poor':
    internet_quality_Poor = 1
    internet_quality_Average = 0
    internet_quality_Good = 0
elif internet_quality.lower() == 'average':
    internet_quality_Poor = 0
    internet_quality_Average = 1
    internet_quality_Good = 0
elif internet_quality.lower() == 'good':
    internet_quality_Poor = 0
    internet_quality_Average = 0
    internet_quality_Good = 1

if extracurricular_participation.lower() == 'no':
    extracurricular_participation = 0
else:
    extracurricular_participation = 1

user_features = np.array([[
    age,
    gender,
    study_hours_per_day,
    social_media_hours,
    netflix_hours,
    part_time_job,
    attendance_percentage,
    sleep_hours,
    exercise_frequency,
    mental_health_rating,
    extracurricular_participation,
    diet_quality_Fair,
    diet_quality_Good,
    diet_quality_Poor,
    parental_education_level_Bachelor,
    parental_education_level_High_School,
    parental_education_level_Master,
    internet_quality_Average,
    internet_quality_Good,
    internet_quality_Poor
]])

columns = [
    'age', 'gender', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'part_time_job', 'attendance_percentage', 'sleep_hours', 'exercise_frequency',
    'mental_health_rating', 'extracurricular_participation',
    'diet_quality_Fair', 'diet_quality_Good', 'diet_quality_Poor',
    'parental_education_level_Bachelor', 'parental_education_level_High School', 'parental_education_level_Master',
    'internet_quality_Average', 'internet_quality_Good', 'internet_quality_Poor'
]

user_features_df = pd.DataFrame(user_features, columns=columns)

user_features_scaled = scaler.transform(user_features_df)



linear_pred = linear_model.predict(user_features_scaled)
rf_pred = rf_model.predict(user_features_scaled)

print("\nPredicted Exam Scores:")
print(f"Linear Regression Prediction: {linear_pred[0]:.2f}")
print(f"Random Forest Regressor Prediction: {rf_pred[0]:.2f}")
