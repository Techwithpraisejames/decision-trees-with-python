import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

st.title('Drug Type Prediction App')
st.write('This app predicts the appropriate drug type based on patient characteristics.')


csv_file_path = "C:\\Users\\HP-PC\\Downloads\\drug200.csv"
drug_prediction_df = pd.read_csv(csv_file_path, encoding='latin1')

# Initialize LabelEncoders for each categorical column and the target
# We'll need separate encoders to map inputs and decode outputs correctly
sex_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
drug_encoder = LabelEncoder()

data_processed = drug_prediction_df.copy()
data_processed['Sex'] = sex_encoder.fit_transform(drug_prediction_df['Sex'])
data_processed['BP'] = bp_encoder.fit_transform(drug_prediction_df['BP'])
data_processed['Cholesterol'] = cholesterol_encoder.fit_transform(drug_prediction_df['Cholesterol'])
data_processed['Drug'] = drug_encoder.fit_transform(drug_prediction_df['Drug'])

# Define features (X) and target (y)
X = data_processed[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data_processed[['Drug']]

# Train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X, y)

# --- Streamlit Sidebar for User Input ---
st.sidebar.header('Input Patient Characteristics')

age = st.sidebar.slider('Age', min_value=15, max_value=75, value=30)
sex_options = sex_encoder.classes_.tolist()
selected_sex_str = st.sidebar.radio('Sex', sex_options)
sex_encoded = sex_encoder.transform([selected_sex_str])[0]

bp_options = bp_encoder.classes_.tolist()
selected_bp_str = st.sidebar.selectbox('Blood Pressure (BP)', bp_options)
bp_encoded = bp_encoder.transform([selected_bp_str])[0]

cholesterol_options = cholesterol_encoder.classes_.tolist()
selected_cholesterol_str = st.sidebar.selectbox('Cholesterol', cholesterol_options)
cholesterol_encoded = cholesterol_encoder.transform([selected_cholesterol_str])[0]

na_to_k = st.sidebar.slider('Na_to_K Ratio', min_value=5.0, max_value=38.0, value=15.0, step=0.1)

# Prediction Section
if st.sidebar.button('Predict Drug Type'):
    input_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex_encoded,
        'BP': bp_encoded,
        'Cholesterol': cholesterol_encoded,
        'Na_to_K': na_to_k
    }])

    prediction_encoded = decision_tree.predict(input_data)
    prediction_drug_name = drug_encoder.inverse_transform(prediction_encoded)[0]

    st.subheader('Prediction Result:')
    st.success(f'Based on the input characteristics, the predicted drug type is: **{prediction_drug_name}**')


# Disclaimer Section
    st.markdown("---")
    st.subheader("Important Disclaimer")
    st.info(
    "This application is intended for educational and research demonstration "
    "purposes only."
    )

