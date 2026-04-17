
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# Store prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

# Page settings
st.set_page_config(
    page_title='CardioPredict',
    page_icon='❤️',
    layout='wide'
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .stButton>button {
        background-color: #e63946;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #c1121f;
        color: white;
    }

    .title-text {
        font-size: 42px;
        font-weight: bold;
        color: #1d3557;
        text-align: center;
    }

    .sub-text {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<p class="title-text">❤️ CardioPredict</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-text">Smart Heart Disease Prediction Using Machine Learning</p>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title('Navigation')

page = st.sidebar.radio(
    'Go to',
    ['Home', 'Prediction', 'About Dataset', 'Model Performance', 'Health Tips']
)

st.sidebar.markdown('---')
st.sidebar.subheader('Project Details')
st.sidebar.write('Project Name: CardioPredict')
st.sidebar.write('Model Used: Random Forest')
st.sidebar.write('Best Accuracy: 90%')
st.sidebar.write('Developed Using: Streamlit + Machine Learning')

# Home Page
if page == 'Home':
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1d3557, #457b9d);
                padding: 30px;
                border-radius: 20px;
                color: white;'>
        <h1 style='text-align:center;'>❤️ Welcome to CardioPredict</h1>
        <p style='text-align:center; font-size:18px;'>
        An intelligent heart disease prediction system powered by Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write('')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Model Used', 'Random Forest')

    with col2:
        st.metric('Best Accuracy', '90%')

    with col3:
        st.metric('Prediction Type', 'Binary Classification')

    st.info('Use the sidebar to explore prediction, dataset details, model performance, and health tips.')

    st.snow()

# Prediction Page
elif page == 'Prediction':
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader('Enter Patient Details')

    with st.expander('Understand the Input Fields'):
        st.write('Age: Risk generally increases with age.')
        st.write('Chest Pain Type: Different chest pain types are linked with different heart conditions.')
        st.write('Cholesterol: High cholesterol may increase heart disease risk.')
        st.write('Blood Pressure: High blood pressure can affect the heart.')
        st.write('Oldpeak: Indicates ST depression during exercise.')

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age', 20, 100, 50)
        sex = st.selectbox('Sex', ['Female', 'Male'])
        cp = st.selectbox(
            'Chest Pain Type',
            ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        )
        trestbps = st.number_input('Resting Blood Pressure', 80, 200, 120)
        chol = st.number_input('Cholesterol Level', 100, 600, 200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.selectbox(
            'Resting ECG Result',
            ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']
        )

    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', 60, 250, 150)
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression (Oldpeak)', 0.0, 10.0, 1.0)
        slope = st.selectbox(
            'Slope of Peak Exercise ST Segment',
            ['Upsloping', 'Flat', 'Downsloping']
        )
        ca = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])
        thal = st.selectbox(
            'Thalassemia',
            ['Normal', 'Fixed Defect', 'Reversible Defect']
        )

    sex = 1 if sex == 'Male' else 0

    cp_mapping = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-anginal Pain': 2,
        'Asymptomatic': 3
    }
    cp = cp_mapping[cp]

    fbs = 1 if fbs == 'Yes' else 0

    restecg_mapping = {
        'Normal': 0,
        'ST-T Wave Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }
    restecg = restecg_mapping[restecg]

    exang = 1 if exang == 'Yes' else 0

    slope_mapping = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }
    slope = slope_mapping[slope]

    thal_mapping = {
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversible Defect': 3
    }
    thal = thal_mapping[thal]

    if st.button('Predict Heart Disease Risk'):
        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        risk_score = probability[0][1]

        st.subheader('Prediction Result')
        st.progress(float(risk_score))
        st.write(f'Prediction Confidence: {risk_score * 100:.2f}%')

        if risk_score < 0.30:
            st.success('🟢 Risk Level: Low')
        elif risk_score < 0.70:
            st.warning('🟠 Risk Level: Moderate')
        else:
            st.error('🔴 Risk Level: High')

        st.markdown(f"""
        <div style='background-color:#ffffff;
                    padding:20px;
                    border-radius:15px;
                    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
                    margin-top:20px;'>
            <h3>Patient Summary</h3>
            <p><b>Age:</b> {age}</p>
            <p><b>Blood Pressure:</b> {trestbps}</p>
            <p><b>Cholesterol:</b> {chol}</p>
            <p><b>Heart Rate:</b> {thalach}</p>
        </div>
        """, unsafe_allow_html=True)

        if prediction[0] == 1:
            st.error('⚠️ High Risk of Heart Disease')
            st.markdown("""
            ### Recommendations
            - Consult a cardiologist
            - Exercise regularly
            - Reduce cholesterol intake
            - Avoid smoking
            - Monitor blood pressure
            """)
            st.snow()
        else:
            st.success('✅ Low Risk of Heart Disease')
            st.markdown("""
            ### Recommendations
            - Continue healthy eating
            - Stay physically active
            - Sleep properly
            - Get regular checkups
            """)
            st.balloons()

        # Save prediction history
        st.session_state.history.append({
            'Age': age,
            'Blood Pressure': trestbps,
            'Cholesterol': chol,
            'Heart Rate': thalach,
            'Risk Score': f'{risk_score * 100:.2f}%',
            'Prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk'
        })

    if st.session_state.history:
        st.subheader('Prediction History')

        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"""
            <div style='background-color:#f8f9fa;
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:10px;
                        border-left:5px solid #457b9d;'>
                <b>Prediction {i}</b><br>
                Age: {item['Age']}<br>
                Blood Pressure: {item['Blood Pressure']}<br>
                Cholesterol: {item['Cholesterol']}<br>
                Heart Rate: {item['Heart Rate']}<br>
                Risk Score: {item['Risk Score']}<br>
                Prediction: {item['Prediction']}
            </div>
            """, unsafe_allow_html=True)

# Dataset Page
elif page == 'About Dataset':
    st.subheader('Dataset Information')

    st.markdown("""
    The dataset contains important medical information such as:

    - Age
    - Sex
    - Chest Pain Type
    - Blood Pressure
    - Cholesterol
    - ECG Results
    - Heart Rate
    - Exercise Induced Angina
    - Number of Major Vessels
    - Thalassemia

    Target Values:
    - 0 = No Heart Disease
    - 1 = Heart Disease Present
    """)

# Model Performance Page
elif page == 'Model Performance':
    st.subheader('Model Accuracy Comparison')

    st.write('Logistic Regression Accuracy: 85%')
    st.progress(0.85)

    st.write('Decision Tree Accuracy: 80%')
    st.progress(0.80)

    st.write('Random Forest Accuracy: 90%')
    st.progress(0.90)

    st.write('Neural Network Accuracy: 88%')
    st.progress(0.88)

    st.success('Random Forest performed best among all models.')

# Health Tips Page
elif page == 'Health Tips':
    st.subheader('Heart Health Tips')

    st.markdown("""
    - Eat healthy food with low cholesterol
    - Exercise regularly
    - Avoid smoking
    - Reduce stress
    - Sleep properly
    - Drink enough water
    - Control blood pressure and sugar levels
    - Go for regular health checkups
    """)

# Footer
st.markdown('---')
st.markdown(
    "<center><h4>'Prevention is better than cure.'</h4></center>",
    unsafe_allow_html=True
)
st.markdown(
    '<center>Developed with Streamlit, Python, and Machine Learning</center>',
    unsafe_allow_html=True
)

