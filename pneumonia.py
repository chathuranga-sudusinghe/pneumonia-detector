import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .pneumonia {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and threshold
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pneumonia_model.h5')
        return model
    except:
        st.error("⚠️ Model file 'pneumonia_model.h5' not found. Please ensure the model is saved in the same directory.")
        return None

@st.cache_data
def load_metadata():
    try:
        class_names = joblib.load("class_names.pkl")
        decision_threshold = joblib.load("decision_threshold.pkl")
        return class_names, decision_threshold
    except:
        st.warning("⚠️ Metadata files not found. Using default values.")
        return ['NORMAL', 'PNEUMONIA'], 0.65

# Prediction function
def predict_image(model, img, decision_threshold):
    # Preprocess image
    img = img.resize((128, 128))
    img_array = np.array(img)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array, verbose=0)
    pred_value = pred[0][0]
    
    return pred_value

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🫁 Pneumonia Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a chest X-ray image to detect pneumonia</p>', unsafe_allow_html=True)
    
    # Load model and metadata
    model = load_model()
    class_names, decision_threshold = load_metadata()
    
    if model is None:
        st.stop()
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a deep learning model to detect pneumonia from chest X-ray images.
        
        **How to use:**
        1. Upload a chest X-ray image
        2. Wait for the prediction
        3. View the results
        
        **Model Details:**
        - Input size: 128x128 pixels
        - Architecture: CNN with data augmentation
        - Classes: Normal, Pneumonia
        """)
        
        st.header("⚙️ Settings")
        decision_threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(decision_threshold),
            step=0.05,
            help="Adjust the threshold for classification"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image in JPG or PNG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded X-ray Image', use_container_width=True)
        
        # Predict button
        if st.button("🔍 Analyze X-ray", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Make prediction
                pred_value = predict_image(model, img, decision_threshold)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Confidence bars
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("NORMAL", f"{(1-pred_value)*100:.1f}%")
                with col2:
                    st.metric("PNEUMONIA", f"{pred_value*100:.1f}%")
                
                # Progress bar
                st.progress(float(pred_value))
                
                # Final prediction
                if pred_value >= decision_threshold:
                    st.markdown(f"""
                    <div class="result-box pneumonia">
                        <h2>⚠️ PNEUMONIA DETECTED</h2>
                        <p style="font-size: 1.1rem;">Confidence: {pred_value*100:.2f}%</p>
                        <p style="font-size: 0.9rem; color: #721c24;">Please consult a healthcare professional for proper diagnosis.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box normal">
                        <h2>✅ NORMAL</h2>
                        <p style="font-size: 1.1rem;">Confidence: {(1-pred_value)*100:.2f}%</p>
                        <p style="font-size: 0.9rem; color: #155724;">No signs of pneumonia detected in this X-ray.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Disclaimer
                st.info("⚕️ **Disclaimer:** This is an AI-assisted tool and should not replace professional medical diagnosis. Always consult with a qualified healthcare provider.")

if __name__ == "__main__":
    main()
