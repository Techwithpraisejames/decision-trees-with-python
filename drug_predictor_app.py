import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(layout="wide")

# App title and description
st.title("Drug Type Prediction App")
st.write(
    "This app predicts the appropriate drug type based on patient characteristics "
    "using a Decision Tree classifier."
)

# Data Loading and Preprocessing
csv_file_path = "data/drug200.csv" # Update with actual path
drug_prediction_df = pd.read_csv(csv_file_path)

# Initialize label encoders
sex_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
drug_encoder = LabelEncoder()

data_processed = drug_prediction_df.copy()
data_processed['Sex'] = sex_encoder.fit_transform(drug_prediction_df['Sex'])
data_processed['BP'] = bp_encoder.fit_transform(drug_prediction_df['BP'])
data_processed['Cholesterol'] = cholesterol_encoder.fit_transform(drug_prediction_df['Cholesterol'])
data_processed['Drug'] = drug_encoder.fit_transform(drug_prediction_df['Drug'])

# Define features and target
X = data_processed[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
y = data_processed["Drug"]

# Train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X, y)

# User Input Section (Main Page)
st.markdown("## Enter Patient Information")

age = st.slider("Age", min_value=15, max_value=75, value=30)
sex_str = st.radio("Sex", sex_encoder.classes_.tolist())
bp_str = st.selectbox("Blood Pressure (BP)", bp_encoder.classes_.tolist())
cholesterol_str = st.selectbox("Cholesterol Level", cholesterol_encoder.classes_.tolist())
na_to_k = st.slider("Na_to_K Ratio", min_value=5.0, max_value=38.0, value=15.0, step=0.1)

# Encode inputs
sex_encoded = sex_encoder.transform([sex_str])[0]
bp_encoded = bp_encoder.transform([bp_str])[0]
cholesterol_encoded = cholesterol_encoder.transform([cholesterol_str])[0]

# Prediction Section
if st.button("Predict Drug Type"):
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex_encoded,
                "BP": bp_encoded,
                "Cholesterol": cholesterol_encoded,
                "Na_to_K": na_to_k,
            }
        ]
    )

    prediction_encoded = decision_tree.predict(input_data)
    prediction_drug_name = drug_encoder.inverse_transform(prediction_encoded)[0]

    st.subheader('Prediction Result:')
    st.success(
        f"The model predicts that the most appropriate drug is **{prediction_drug_name}**."
    )

    # Disclaimer
    st.markdown("---")
    st.subheader("Important Disclaimer")
    st.info(
        "This application is intended for educational and research demonstration "
        "purposes only."
    )

