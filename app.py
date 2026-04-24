import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import DepthwiseConv2D
import cv2

# Custom DepthwiseConv2D to handle legacy models with 'groups' parameter
class LegacyDepthwiseConv2D(DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        # Remove 'groups' parameter if present (not supported in Keras 3.x)
        config.pop('groups', None)
        return super().from_config(config)

# Helper function to load legacy models safely
def safe_load_model(model_path):
    """Load model with custom objects to handle legacy compatibility issues"""
    try:
        # For malaria model, try multiple loading strategies
        if 'malaria' in model_path.lower():
            # Try 1: Standard loading without custom objects
            try:
                return load_model(model_path, compile=False)
            except:
                pass
            
            # Try 2: Legacy H5 format loading
            try:
                from tensorflow.keras.saving import load_model as legacy_load
                return legacy_load(model_path, compile=False, safe_mode=False)
            except:
                pass
            
            # Try 3: With custom objects
            try:
                custom_objects = {'DepthwiseConv2D': LegacyDepthwiseConv2D}
                return load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e:
                st.error(f"Could not load malaria model. The model file may be corrupted.")
                st.info("Please retrain the malaria model or use a compatible model file.")
                return None
        else:
            # For other models, use custom objects
            custom_objects = {'DepthwiseConv2D': LegacyDepthwiseConv2D}
            return load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

# CSS for custom styling
def local_css():
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #e0e0e0;
    }
    .highlight {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .stApp {
        background-color: #0d1117;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    .disease-title {
        color: #58a6ff;
        font-weight: bold;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    /* Text color */
    .css-10trblm {
        color: #c9d1d9;
    }
    /* Radio button text */
    .st-emotion-cache-16idsys p {
        color: #c9d1d9;
    }
    </style>
    """, unsafe_allow_html=True)

class DiseaseDetector:
    @staticmethod
    def detect_alzheimers(file):
        model = safe_load_model('ALZ.h5')
        if model is None:
            return "Error loading model"
        img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
        i = img_to_array(img)
        i = i / 255.0
        input_arr = np.expand_dims(i, axis=0)
        pred = np.argmax(model.predict(input_arr), axis=-1)
        
        labels = {
            0: "Mild Demented",
            1: "Moderate Demented", 
            2: "Non Demented", 
            3: "Very Mild Demented"
        }
        return labels.get(pred[0], "Unknown")

    @staticmethod
    def detect_brain_tumor(file):
        model = safe_load_model('BR.h5')
        if model is None:
            return "Error loading model"
        IMAGE_SIZE = 150
        image_obj = Image.open(file)
        image_array = np.array(image_obj)
        image_resized = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
        images = image_resized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        predictions = model.predict(images)
        labels = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']
        return labels[np.argmax(predictions, axis=1)[0]]

    @staticmethod
    def detect_pneumonia(file):
        loaded_model = safe_load_model('PN.h5')
        if loaded_model is None:
            return "Error loading model"
        image1 = tf.keras.preprocessing.image.load_img(file, target_size=(150, 150))
        image1 = img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        img_array = image1 / 255.0 
        prediction = loaded_model.predict(img_array)
        if prediction[0][0] > 0.3:
            return "Pneumonia Detected"
        else:
            return "No Pneumonia Detected"





def home_page():
    st.title("🏥 Healthify: Advanced Disease Detection Platform")
    
    st.markdown("""
    <div class="highlight">
    <h2 class="disease-title">Welcome to Healthify</h2>
    <p class="big-font">
    Healthify is an advanced medical image analysis platform that leverages 
    cutting-edge machine learning to detect various diseases quickly and accurately.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Our Detection Capabilities:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🧠 Alzheimer's")
        st.markdown("Detect early stages of dementia")
    
    with col2:
        st.markdown("#### 🧬 Brain Tumor")
        st.markdown("Identify potential brain abnormalities")
    
    with col3:
        st.markdown("#### 🫁 Pneumonia")
        st.markdown("Analyze chest X-rays for lung infections")

def brain_tumor_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🧠 Brain Tumor Detection")
    with col2:
        try:
            st.image("brainimg.png", width=200)
        except:
            pass  # Skip if image not found
    
    st.markdown("""
    <div class="highlight">
    <h3>Brain Tumor Detection using Machine Learning</h3>
    <p class="big-font">
    Our advanced neural network analyzes brain MRI scans to detect potential tumors.
    Supports detection of No Tumor, Pituitary, Meningioma, and Glioma Tumors.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        
        if st.button("Detect Brain Tumor"):
            with st.spinner('Analyzing MRI...'):
                result = DiseaseDetector.detect_brain_tumor(uploaded_file)
                st.success(f"Brain Tumor Prediction: {result}")

def pneumonia_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🫁 Pneumonia Detection")
    with col2:
        try:
            st.image("pne.png", width=200)
        except:
            pass  # Skip if image not found
    
    st.markdown("""
    <div class="highlight">
    <h3>Pneumonia Detection from Chest X-Rays</h3>
    <p class="big-font">
    Our AI model can detect pneumonia by analyzing chest radiographs 
    with high accuracy and speed.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded X-Ray", width=300)
        
        if st.button("Detect Pneumonia"):
            with st.spinner('Analyzing X-Ray...'):
                result = DiseaseDetector.detect_pneumonia(uploaded_file)
                st.success(result)



def alzheimer_page():
    col1, col2 = st.columns([7,3])
    with col1:
        st.title("🧠 Alzheimer's Detection")
    with col2:
        try:
            st.image("az.png", width=200)
        except:
            pass  # Skip if image not found
    
    st.markdown("""
    <div class="highlight">
    <h3>Early Alzheimer's Stage Detection</h3>
    <p class="big-font">
    Machine learning model to detect different stages of Alzheimer's 
    from brain MRI scans.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Brain MRI", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        
        if st.button("Detect Alzheimer's Stage"):
            with st.spinner('Analyzing Brain Scan...'):
                result = DiseaseDetector.detect_alzheimers(uploaded_file)
                st.success(f"Alzheimer's Status: {result}")



def exit_page():
    st.title("👋 Thank You for Using Healthify")
    st.markdown("""
    <div class="highlight">
    <h3>Your Health, Our Priority</h3>
    <p class="big-font">
    Healthify is committed to using advanced AI technologies 
    to support early disease detection and healthcare diagnostics.
    
    Remember: Our AI assists medical professionals, 
    but does not replace professional medical advice.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Exit Button with Balloons
    if st.button("Exit Healthify"):
        st.balloons()
        st.stop()

def main():
    st.set_page_config(page_title="Healthify", page_icon="🏥", layout="wide")  # Move this line here
    local_css()

    st.sidebar.title("Healthify")

    # Add an image in the sidebar (commented out if image doesn't exist)
    try:
        st.sidebar.image("images.jpeg", width=250)
    except:
        st.sidebar.markdown("### 🏥 Healthify")  # Fallback if image not found
    
    # Pages dictionary
    pages = {
        "Home": home_page,
        "Brain Tumor Detection": brain_tumor_page,
        "Pneumonia Detection": pneumonia_page,
        "Alzheimer's Detection": alzheimer_page,
        "Exit": exit_page
    }
    # Sidebar navigation
    page = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[page]()

    # Update sidebar with developer info and college
    st.sidebar.markdown("### 👤 About")

    st.sidebar.info(
    """
    **Developers:**  
    Varshini T  
    Mahima L  
    Vijayalakshmi R  

    **College:**  
    Government College of Engineering, Tirunelveli 🎓
    """
    )

# Run the app
if __name__ == "__main__":
    main()