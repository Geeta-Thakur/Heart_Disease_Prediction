Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
Load the dataset
def load_data():
    data = pd.read_csv('heart.csv')
    return data
Preprocess the data
def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
Make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions
Evaluate the model
def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report
Map categorical values to numerical values
def map_values(sex, cp, fbs, restecg, exang, ca, thal):
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'< 120 mg/dl': 0, '>= 120 mg/dl': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'Yes': 1, 'No': 0}
    ca_map = {'0': 0, '1': 1, '2': 2, '3': 3}
    thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
    sex = sex_map[sex]
    cp = cp_map[cp]
    fbs = fbs_map[fbs]
    restecg = restecg_map[restecg]
    exang = exang_map[exang]
ca = ca_map[ca]
    thal = thal_map[thal]
    return sex, cp, fbs, restecg, exang, ca, thal
Create the Streamlit app
def create_app():
    st.title("Heart Disease Prediction")
    st.write("This app predicts the likelihood of heart disease based on several factors.")
    # Load the data
    data = load_data()
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    # Train the model
    model = train_model(X_train, y_train)
    # Make predictions
    predictions = make_predictions(model, X_test)
    # Evaluate the model
    accuracy, report = evaluate_model(y_test, predictions)
    # Display the results
st.write("Model Accuracy:", accuracy)
    st.write("Classification Report:")
    st.write(report)
    # Create a form for user input
    st.write("Enter your details to get a prediction:")
    age = st.number_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol Level")
    fbs = st.selectbox("Fasting Blood Sugar", ["< 120 mg/dl", ">= 120 mg/dl"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", ["0", "1", "2", "3"])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
Map categorical values to numerical values
sex, cp, fbs, restecg, exang, ca, thal = map_values(sex, cp, fbs, restecg, exang, ca, thal)
Create a DataFrame for user input
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})
Make a prediction using the trained model
prediction = model.predict(user_input)
Display the prediction result
if st.button("Get Prediction"):
    if prediction[0] == 1:
        st.write("You are likely to have heart disease.")
    else:
        st.write("You are unlikely to have heart disease.")
Run the app
if name == "main":
create_app()
