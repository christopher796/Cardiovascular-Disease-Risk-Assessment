import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

data_path = "CVD Dataset.csv"

dataset = pd.read_csv(data_path)

cvd_dataset = dataset.dropna(axis = 0)

sex_map = {'M': 0, 'F': 1}
smoking_map = {'N': 0, 'Y': 1}
diabetes_map = {'N': 0, 'Y': 1}
physical_map = {'Low': 0, 'Moderate': 1, 'High': 2}
family_map = {'N': 0, 'Y': 1}
bpc_map = {'Hypertension Stage 1': 0, 'Normal': 1, 'Hypertension Stage 2': 2, 'Elevated': 3}
risk_map = {'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2}
inverse_risk_map = {v: k for k, v in risk_map.items()}

cvd_dataset['Sex'] = cvd_dataset['Sex'].map(sex_map)
cvd_dataset['Smoking Status'] = cvd_dataset['Smoking Status'].map(smoking_map)
cvd_dataset['Diabetes Status'] = cvd_dataset['Diabetes Status'].map(diabetes_map)
cvd_dataset['Physical Activity Level'] = cvd_dataset['Physical Activity Level'].map(physical_map)
cvd_dataset['Family History of CVD'] = cvd_dataset['Family History of CVD'].map(family_map)
cvd_dataset['Blood Pressure Category'] = cvd_dataset['Blood Pressure Category'].map(bpc_map)
cvd_dataset['CVD Risk Level'] = cvd_dataset['CVD Risk Level'].map(risk_map)

X = cvd_dataset[['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Abdominal Circumference (cm)', 'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 'Height (cm)',
'Waist-to-Height Ratio', 'Systolic BP', 'Diastolic BP', 'Blood Pressure Category', 'Estimated LDL (mg/dL)']]

y_score = cvd_dataset['CVD Risk Score']

y_level = cvd_dataset['CVD Risk Level']

X_train, X_test, y_score_train, y_score_test = train_test_split(X, y_score, test_size = 0.2, random_state = 42)

X_train, X_test, y_level_train, y_level_test = train_test_split(X, y_level, test_size = 0.2, random_state = 42)

reg_model = RandomForestRegressor(random_state = 42)
reg_model.fit(X_train, y_score_train)

clf_model = RandomForestClassifier(random_state = 42)
clf_model.fit(X_train, y_level_train)

score_pred = reg_model.predict(X_test)
level_pred = clf_model.predict(X_test)

print("Regression MAE: ", mean_absolute_error(y_score_test, score_pred))
print("Classification Accuracy: ", accuracy_score(y_level_test, level_pred))

Sex = input("Enter Sex (M/F): ").upper()
Age = int(input("Enter your Age: "))
Weight = float(input("Enter your Weight(Kg): "))
Height = float(input("Enter your Height(m): "))
BMI = float(input("Enter BMI: "))
ABS = float(input("Enter Abdominal Circumference(cm): "))
Cholesterol = int(input("Enter Total Cholesterol(mg/dL): "))
HDL = int(input("Enter HDL(mg/dL): "))
Fasting = int(input("Enter Fasting Blood Sugar(mg/dL): "))
Smoking = input("Do you Smoke(Y/N): ").upper()
Diabetes = input("Are you Diabetic(Y/N): ").upper()
Physical = input("Physical Activity Level(Low/Moderate/High): ")
Family = input("Any CVD case in your family: ")
Height = int(input("Enter your Height(cm): "))
Ratio = float(input("Enter Waist-to-Height Ratio: "))
Systolic = int(input("Enter Systolic BP: "))
Diastolic = int(input("Enter Diastolic BP: "))
BPC = input("Enter Blood Pressure Category(Hypertension Stage 1/Normal/Hypertension Stage 2/ Elevated): ")
LDL = int(input("Enter Estimated LDL(mg/dL): "))
sex_encoded = sex_map[Sex]
smoking_encoded = smoking_map[Smoking]
diabetes_encoded = diabetes_map[Diabetes]
physical_encoded = physical_map[Physical]
family_encoded = family_map[Family]
bpc_encoded = bpc_map[BPC]

user_input = [[sex_encoded, Age, Weight, Height, BMI, ABS, Cholesterol, HDL, Fasting, smoking_encoded,
diabetes_encoded, physical_encoded, family_encoded, Height, Ratio, Systolic, Diastolic, bpc_encoded,
LDL]]

predicted_score = reg_model.predict(user_input)[0]

predicted_level_num = clf_model.predict(user_input)[0]
predicted_level = inverse_risk_map[predicted_level_num]

print("\n --- RESULTS ---")
print(f" CVD Risk Score: {predicted_score: .2f}")
print(f"CVD Risk Level: {predicted_level}")

