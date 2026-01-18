CARDIOVASCULAR DISEASE (CVD) RISK PREDICTION MODEL.
This project is a machine learning-based CVD risk prediction system that estimates both:
1. CVD Risk Score(Numerical).
2. CVD Risk Level (C ategorical: LOW/ INTERMEDIARY/ HIGH).
The model accepts mixed data types (categorical + numerical) and is designed to be interactive, prompting users for inputs and returning interpretable medical risk outputs.

PROBLEM STATEMENT
Cardiovascular diseases are among the leading causes of death globally. Early risk assessment helps in preventive healthcare decision-making.

The project aims to:
1. Predict a continous CVD risk score.
2. Classify patients into risk levels.
3. Handle real-world medical features such as sex and blood pressure.

MODELS USED
1. RandomForestRegressor for Risk Score Prediction.
2. RandomForestClassifier for Risk Level Classification.
Regression and classification are handled using separate models, which is best practice in ML.

MODEL EVALUATION
REGRESSION PERFORMANCE
1. METRIC: Mean Absolute Error(MAE).
2. RESULT: MAE = 0.4
3. Target Range: 0 - 20

CLASSIFICATION PERFOMANCE
1. METRIC: Accuracy.
2. RESULT: 60%

USER INTERACTION
The model allows command-line user input, making it easy to test predictions interactively.
The model internally uses numeric encoding, then converts predictions back to human-readable labels.

TECHNOLOGIES USED
1. Python
2. Pandas
3. Scikit-learn
4. Random Forest Algorithms

FUTURE IMPROVEMENTS
1. Use pipelines & columnTransformer
2. Deploy as a web app(FLASK/FastAPI)
3. Improve classification using risk-score-based thresholds.
4. Add cross-validation and hyperparameter tuning.
