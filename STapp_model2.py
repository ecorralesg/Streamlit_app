import streamlit as st
import pickle
import numpy as np

# Load the trained models and their accuracies
def load_model(dataset_name, model_name):
    filename = f"{model_name.replace(' ', '_').lower()}_{dataset_name}_model.pkl"
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data['model'], data['accuracy']

# Main function to run the app
def main():
    st.markdown(
        '<style>h1{white-space: nowrap;}</style>',
        unsafe_allow_html=True
    )
    
    st.title('🚀 Machine Learning Classifier Web App 🚀')

    # Select dataset and model
    dataset_name = st.sidebar.selectbox('Select Dataset', ['bank_retirement', 'diabetes'])
    model_name = st.sidebar.selectbox('Select Model', ['Logreg', 'SVM'])

    # Load the trained model and its accuracy based on user's selections
    model, model_accuracy = load_model(dataset_name, model_name)

    # Input form to get features from the user
    st.header(f'🔍 Enter your inputs for {dataset_name.capitalize()}🔍')

    if dataset_name == 'bank_retirement':
        age = st.slider('Age', 20, 100, 50)
        savings_401k = st.slider('401K Savings ($)', 0, 1000000, 200000)
        features = np.array([age, savings_401k]).reshape(1, -1)

    elif dataset_name == 'diabetes':
        pregnancies = st.slider('Pregnancies', 0, 17, 3)
        glucose = st.slider('Glucose', 0, 199, 117)
        blood_pressure = st.slider('Blood Pressure', 0, 122, 72)
        skin_thickness = st.slider('Skin Thickness', 0, 99, 23)
        insulin = st.slider('Insulin', 0, 846, 30)
        bmi = st.slider('BMI', 0.0, 67.1, 32.0)
        diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
        age = st.slider('Age', 21, 81, 29)

        features = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)

        # Load the saved scaler and scale the input features for diabetes (if you have a scaler for it)
        with open('diabetes_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            features = scaler.transform(features)

    # Predict button
    if st.button('Predict'):
        prediction = model.predict(features)

        # Display the prediction result with an animation
        if dataset_name == 'bank_retirement':
            bank_retirement_classes = ['Not able to retire', 'Can retire']
            prediction_label = bank_retirement_classes[prediction[0]]
            st.success(f'Prediction: {prediction_label}')
        elif dataset_name == 'diabetes':
            diabetes_classes = ['Negative', 'Positive']
            prediction_label = diabetes_classes[prediction[0]]
            st.success(f'Prediction: Diabetes Status - {prediction_label}')

        # Display the model's accuracy
        st.info(f'🎯 Model Accuracy: {model_accuracy:.2f}')

        # Add some balloons for fun!
        st.balloons()

if __name__ == '__main__':
    main()
