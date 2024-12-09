import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_data():
    try:
        data = pd.read_csv('heart.csv')
        return data
    except FileNotFoundError:
        st.error("Heart dataset not found. Please ensure 'heart.csv' is in the correct directory.")
        return None

# Preprocess the data
def preprocess_data(data):
    if data is None:
        return None, None, None, None
    
    # Ensure 'slope' is properly encoded in the original dataset
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    data['slope'] = data['slope'].map(slope_map)
    
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    if X_train is None or y_train is None:
        return None
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Make predictions
def make_predictions(model, X_test):
    if model is None or X_test is None:
        return None
    
    predictions = model.predict(X_test)
    return predictions

# Evaluate the model
def evaluate_model(y_test, predictions):
    if y_test is None or predictions is None:
        return None, None
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Map categorical values to numerical values
def map_values(sex, cp, fbs, restecg, exang, ca, thal, slope):
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'< 120 mg/dl': 0, '>= 120 mg/dl': 1}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'Yes': 1, 'No': 0}
    ca_map = {'0': 0, '1': 1, '2': 2, '3': 3}
    thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}

    return (
        sex_map[sex],
        cp_map[cp],
        fbs_map[fbs],
        restecg_map[restecg],
        exang_map[exang],
        ca_map[ca],
        thal_map[thal],
        slope_map[slope]
    )

# Create the Streamlit app
def create_app():
    st.title("Heart Disease Prediction")
    st.write("This app predicts the likelihood of heart disease based on several factors.")
    
    # Load the data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check the dataset.")
        return

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    if X_train is None:
        st.error("Failed to preprocess data.")
        return

    # Train the model
    model = train_model(X_train, y_train)
    if model is None:
        st.error("Failed to train the model.")
        return

    # Make predictions
    predictions = make_predictions(model, X_test)
    if predictions is None:
        st.error("Failed to make predictions.")
        return

    # Evaluate the model
    accuracy, report = evaluate_model(y_test, predictions)
    if accuracy is not None:
        st.write("Model Accuracy:", accuracy)
        st.write("Classification Report:")
        st.write(report)

    # Create a form for user input
    st.write("Enter your details to get a prediction:")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
    chol = st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar", ["< 120 mg/dl", ">= 120 mg/dl"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", ["0", "1", "2", "3"])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Map categorical values to numerical values
    sex, cp, fbs, restecg, exang, ca, thal, slope = map_values(sex, cp, fbs, restecg, exang, ca, thal, slope)

    # Create a DataFrame for user input
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

    # Prediction button
    if st.button("Get Prediction"):
        # Make a prediction using the trained model
        prediction = model.predict(user_input)
        
        # Display the prediction result
        if prediction[0] == 1:
            st.warning("You are likely to have heart disease.")
        else:
            st.success("You are unlikely to have heart disease.")

# Run the app
if __name__ == "__main__":
    create_app()
