import streamlit as st
from model import train_pipeline, predict_patient

# Page Configuration
st.set_page_config(
    page_title="Orthopedic Patient Classifier",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Biomechanical Orthopedic Prediction")
st.write("This application utilizes Machine Learning to classify spine conditions into **Normal**, **Disk Hernia**, or **Spondylolisthesis** based on biomechanical patient features.")

# Cache the model training process so it doesn't retrain on every slider adjustment
@st.cache_resource
def get_trained_model():
    return train_pipeline()

with st.spinner("Downloading dataset and training AI model..."):
    model, scaler, accuracy = get_trained_model()

# Sidebar metric dashboard


# User Input Form
st.header("Patient Biomechanical Measurements")
st.write("Adjust the sliders below based on the patient's radiograph / MRI metrics:")

# Layout sliders in two parallel columns
col1, col2 = st.columns(2)

with col1:
    pelvic_incidence = st.slider("Pelvic Incidence", 26.0, 130.0, 60.0, 0.1)
    pelvic_tilt = st.slider("Pelvic Tilt", -6.0, 49.0, 17.0, 0.1)
    lumbar_lordosis = st.slider("Lumbar Lordosis Angle", 14.0, 125.0, 50.0, 0.1)

with col2:
    sacral_slope = st.slider("Sacral Slope", 13.0, 121.0, 43.0, 0.1)
    pelvic_radius = st.slider("Pelvic Radius", 70.0, 163.0, 117.0, 0.1)
    spondylolisthesis = st.slider("Degree of Spondylolisthesis", -11.0, 418.0, 25.0, 0.1)

# Compile inputs
input_data = [
    pelvic_incidence,
    pelvic_tilt,
    lumbar_lordosis,
    sacral_slope,
    pelvic_radius,
    spondylolisthesis
]

st.markdown("---")

# Prediction Execution
if st.button("Analyze Biomechanical Data", type="primary"):
    prediction, probabilities = predict_patient(model, scaler, input_data)
    
    st.header("AI Diagnostic Outcome")
    
    # Dynamic styling based on prediction outcome
    if prediction == "Normal":
        st.success(f"Prediction: **{prediction}**")
    elif prediction == "Disk Hernia":
        st.warning(f"Prediction: **{prediction}**")
    else:
        st.error(f"Prediction: **{prediction}**")
        
    # Show confidence metrics
    st.subheader("Prediction Confidence Breakdown")
    classes = model.classes_
    for cls, prob in zip(classes, probabilities):
        st.write(f"- **{cls}**: {prob * 100:.1f}%")