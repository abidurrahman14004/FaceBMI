import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd
import io
import sys

# Add the current directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bmi_predictor import BMIPredictor, MEDIAPIPE_AVAILABLE, MEDIAPIPE_IMPORT_ERROR

# Page configuration
st.set_page_config(
    page_title="BMI Predictor from Face Image",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 5px;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .bmi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .category {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .underweight { color: #3498db; }
    .normal { color: #2ecc71; }
    .overweight { color: #f39c12; }
    .obese { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_predictor():
    """Load the BMI predictor model (cached)"""
    try:
        model_path = os.path.join('models', 'hybrid_model_v2.pth')
        
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at {model_path}"
            st.warning(f"⚠️ {error_msg}")
            return None, error_msg
        
        predictor = BMIPredictor(model_path=model_path)
        
        # Check if model loaded successfully
        if predictor.model_loaded:
            return predictor, None
        else:
            # Return error message, or default message if None
            error_msg = predictor.load_error or "Model failed to load. Please check MediaPipe installation and model file."
            return None, error_msg
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in load_predictor: {error_details}")
        return None, f"Error initializing predictor: {str(e)}"

def get_bmi_category_class(category):
    """Get CSS class for BMI category"""
    category_lower = category.lower()
    if 'underweight' in category_lower:
        return 'underweight'
    elif 'normal' in category_lower:
        return 'normal'
    elif 'overweight' in category_lower:
        return 'overweight'
    elif 'obese' in category_lower:
        return 'obese'
    return ''

def predict_bmi(image_bytes, predictor):
    """Predict BMI from image bytes using the model"""
    try:
        # Validate predictor
        if predictor is None:
            return {
                'success': False,
                'error': 'Model predictor is not initialized. Please reload the page.'
            }
        
        # Ensure model is loaded
        if not predictor.model_loaded:
            return {
                'success': False,
                'error': 'Model is not loaded. Please check model file.'
            }
        
        # Call the model's predict method - THIS USES YOUR MODEL
        result = predictor.predict(image_bytes)
        
        # Validate result
        if not isinstance(result, dict):
            return {
                'success': False,
                'error': 'Invalid result format from model prediction.'
            }
        
        return result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict_bmi: {error_details}")
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }

def main():
    # Sidebar
    with st.sidebar:
        st.title("🏥 BMI Predictor")
        st.markdown("### *Research-Driven AI Application*")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "ℹ️ About", "🔮 Prediction App", "📊 Sample Images", "🔒 Privacy Policy"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
        ### Quick Info
        This application uses deep learning to predict BMI from facial images.
        
        **Led by:**  
        **Dr. Amith Khandakar**  
        Qatar Research Team
        
        ### Supported Formats:
        - PNG, JPG, JPEG, GIF, WEBP
        - Max file size: 16MB
        """)
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_landing_page()
    elif page == "ℹ️ About":
        show_about_page()
    elif page == "🔮 Prediction App":
        show_prediction_app()
    elif page == "📊 Sample Images":
        show_samples_page()
    else:
        show_privacy_policy()

def show_landing_page():
    """Landing page with research overview and team information"""
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; color: #2c3e50; margin-bottom: 0.5rem;">
            🏥 BMI Prediction from Facial Images
        </h1>
        <h3 style="color: #7f8c8d; font-weight: 400;">
            Advanced AI-Powered Body Mass Index Estimation
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Research Leadership
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white; text-align: center;">
        <h2 style="margin: 0; color: white;">Research Led By</h2>
        <h1 style="margin: 0.5rem 0; color: white; font-size: 2.5rem;">Dr. Amith Khandakar</h1>
        <h3 style="margin: 0; color: #f0f0f0;">Qatar Research Team</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Research Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔬 Research Overview")
        st.markdown("""
        This application represents cutting-edge research in non-invasive health assessment using 
        artificial intelligence. Our hybrid deep learning model combines facial image analysis with 
        anthropometric features to predict Body Mass Index (BMI) with high accuracy.
        
        **Key Features:**
        - 🤖 **Multimodal Deep Learning**: Combines facial images and anthropometric features
        - 🎯 **High Accuracy**: Research-validated prediction model
        - ⚡ **Real-Time Processing**: Instant BMI estimation
        - 🔒 **Privacy-First**: No data storage, immediate processing
        - 📱 **Accessible**: Web-based interface for easy access
        
        **Applications:**
        - Remote health monitoring
        - Preliminary health screening
        - Research and educational purposes
        - Telehealth applications
        """)
    
    with col2:
        st.markdown("### 👥 Research Team")
        st.markdown("""
        **Principal Investigator:**
        
        **Dr. Amith Khandakar**  
        Qatar Research Team
        
        ---
        
        **Research Focus:**
        - AI in Healthcare
        - Computer Vision
        - Biomedical Engineering
        - Non-invasive Diagnostics
        
        ---
        
        **Institution:**  
        Qatar University  
        Research & Development
        """)
    
    st.markdown("---")
    
    # Model Performance Highlights
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: #2ecc71; margin: 0;">0.5923</h2>
            <p style="color: #27ae60; margin: 0.5rem 0 0 0;">R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: #3498db; margin: 0;">1.44</h2>
            <p style="color: #2980b9; margin: 0.5rem 0 0 0;">MAE (kg/m²)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: #f39c12; margin: 0;">35.2%</h2>
            <p style="color: #e67e22; margin: 0.5rem 0 0 0;">Within ±1 BMI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background-color: #fce4ec; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: #e74c3c; margin: 0;">81.2%</h2>
            <p style="color: #c0392b; margin: 0.5rem 0 0 0;">Within ±3 BMI</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it Works
    st.markdown("### ⚙️ How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
            <h4>1. Upload Image</h4>
            <p style="color: #7f8c8d;">Upload a clear frontal facial photograph</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
            <h4>2. AI Analysis</h4>
            <p style="color: #7f8c8d;">Deep learning model analyzes facial features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h4>3. Get Results</h4>
            <p style="color: #7f8c8d;">Receive instant BMI prediction and category</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call to Action
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">Ready to Try?</h3>
        <p style="color: #7f8c8d; font-size: 1.1rem; margin-bottom: 1.5rem;">
            Navigate to the <strong>Prediction App</strong> to start analyzing facial images for BMI estimation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Important Notes
    st.info("**📌 Important:** This tool is for research and educational purposes only. Always consult healthcare professionals for medical advice and accurate health assessments.")

def show_about_page():
    """Detailed about page with research methodology"""
    st.title("About Our Research")
    
    st.markdown("---")
    
    # Research Background
    st.markdown("### 🎓 Research Background")
    st.markdown("""
    Body Mass Index (BMI) is a crucial health indicator used worldwide to assess body composition 
    and potential health risks. Traditional BMI calculation requires direct measurement of height 
    and weight, which can be challenging in remote or resource-limited settings.
    
    Our research, led by **Dr. Amith Khandakar** and the **Qatar Research Team**, explores the feasibility 
    of using facial images as a non-invasive alternative for BMI estimation, leveraging advances in 
    computer vision and deep learning.
    """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("### 🔬 Methodology")
    
    tab1, tab2, tab3 = st.tabs(["Model Architecture", "Training Process", "Validation"])
    
    with tab1:
        st.markdown("""
        #### Hybrid Deep Learning Architecture
        
        Our model employs a multimodal approach combining:
        
        1. **Image Processing Branch**
           - Custom Convolutional Neural Network for facial feature extraction
           - Fine-tuned on BMI-specific facial characteristics
           - Advanced feature learning through deep hierarchical representations
        
        2. **Tabular Features Branch**
           - Anthropometric measurements (face dimensions, ratios)
           - Demographic information (age, sex)
           - Engineered features (face width-to-height ratio, symmetry metrics)
        
        3. **Fusion Layer**
           - Multi-head attention mechanism
           - Combines visual and tabular representations
           - Adaptive feature weighting for optimal predictions
        
        4. **Prediction Head**
           - Multi-task learning (BMI, age, sex, BMI category)
           - Uncertainty-aware predictions
           - Comprehensive regularization for generalization
        """)
    
    with tab2:
        st.markdown("""
        #### Training Strategy
        
        **Dataset:**
        - 24,000 training samples
        - 6,000 validation samples
        - Diverse demographic representation
        
        **Data Augmentation:**
        - Random rotation and horizontal flipping
        - Color jittering for lighting variations
        - Random cropping and resizing
        - Feature noise injection
        
        **Optimization:**
        - AdamW optimizer with weight decay
        - Cosine annealing learning rate schedule
        - Early stopping with patience
        - 5-fold cross-validation
        
        **Regularization:**
        - Dropout (0.3)
        - Batch normalization
        - Gradient clipping
        - L2 weight decay
        """)
    
    with tab3:
        st.markdown("""
        #### Validation & Testing
        
        **Ablation Study Results:**
        - Tested 10 different model configurations
        - Evaluated impact of each component
        - Tabular features alone: R² = 0.3387
        - Images alone: R² = 0.0488
        - Combined (full model): R² = 0.2592
        
        **Key Findings:**
        - Tabular features provide stronger predictive signal than images alone
        - Multimodal fusion improves model robustness
        - Data augmentation and regularization prevent overfitting
        - Custom CNN architecture captures facial patterns effectively
        
        **Performance Metrics:**
        - Mean Absolute Error (MAE): 1.78 kg/m²
        - Root Mean Squared Error (RMSE): 2.25 kg/m²
        - Prediction accuracy within ±3 BMI units: 81.2%
        """)
    
    st.markdown("---")
    
    # Research Team
    st.markdown("### 👨‍🔬 Research Team")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">👨‍💼</div>
            <h3 style="margin: 0;">Dr. Amith Khandakar</h3>
            <p style="color: #7f8c8d; margin-top: 0.5rem;">Principal Investigator</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **Dr. Amith Khandakar** leads this groundbreaking research at Qatar University, 
        focusing on applying artificial intelligence and machine learning techniques to 
        biomedical engineering challenges.
        
        **Research Expertise:**
        - Artificial Intelligence in Healthcare
        - Biomedical Signal and Image Processing
        - Machine Learning for Medical Diagnostics
        - Non-invasive Health Monitoring Systems
        
        **Qatar Research Team** comprises dedicated researchers and engineers working on 
        innovative solutions for healthcare challenges using cutting-edge AI technologies.
        """)
    
    st.markdown("---")
    
    # Publications & Citations
    st.markdown("### 📚 Research Impact")
    
    st.markdown("""
    This research contributes to the growing body of work on non-invasive health assessment 
    methods using artificial intelligence. The findings demonstrate the potential of multimodal 
    deep learning approaches for BMI estimation from facial images.
    
    **Research Contributions:**
    - Novel hybrid architecture combining CNN and tabular data
    - Comprehensive ablation study analyzing model components
    - Validation on diverse demographic dataset
    - Open-source implementation for research community
    """)

def show_prediction_app():
    """Main prediction page"""
    st.title("BMI Prediction from Face Image")
    st.markdown("Upload a clear frontal face image to predict BMI")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            predictor, error = load_predictor()
            
            if predictor is not None:
                st.session_state.predictor = predictor
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                
                # Show model information
                with st.expander("ℹ️ Model Information"):
                    st.write(f"**Model:** `hybrid_model_v2.pth`")
                    st.write(f"**Model Path:** `{predictor.model_path}`")
                    st.write(f"**Device:** {predictor.device}")
                    st.write(f"**Model Status:** {'✅ Loaded' if predictor.model_loaded else '❌ Not Loaded'}")
                    st.write(f"**Feature Extraction:** Real features from image (NO dummy data)")
                    if hasattr(predictor, 'feature_cols') and predictor.feature_cols:
                        st.write(f"**Number of Features:** {len(predictor.feature_cols)}")
                    if hasattr(predictor, 'num_landmarks'):
                        st.write(f"**Landmarks:** {predictor.num_landmarks}")
            else:
                # Display error message
                error_display = error if error else "Unknown error occurred during model loading"
                st.error(f"❌ Failed to load model: {error_display}")
                
                # Check if it's a MediaPipe error
                error_str = str(error) if error else ""
                if "MediaPipe" in error_str or "mediapipe" in error_str.lower():
                    st.error("""
                    **❌ MediaPipe Not Available**
                    
                    MediaPipe is required for facial landmark extraction but is not installed.
                    
                    **Installation:**
                    ```bash
                    pip install mediapipe==0.10.13
                    ```
                    
                    Or install all requirements:
                    ```bash
                    pip install -r requirements.txt
                    ```
                    
                    After installation, please restart the Streamlit app.
                    """)
                    
                    # Show diagnostic information
                    with st.expander("🔍 Diagnostic Information"):
                        st.write("**MediaPipe Status:**")
                        st.write(f"- Available: {MEDIAPIPE_AVAILABLE}")
                        if MEDIAPIPE_IMPORT_ERROR:
                            st.write(f"- Error: {MEDIAPIPE_IMPORT_ERROR}")
                        
                        # Try to check if MediaPipe can be imported
                        st.write("\n**Testing MediaPipe Import:**")
                        try:
                            import mediapipe as mp
                            st.success(f"✅ MediaPipe can be imported")
                            if hasattr(mp, '__version__'):
                                st.write(f"   Version: {mp.__version__}")
                            
                            # Check new API
                            try:
                                from mediapipe.python.solutions.face_mesh import FaceMesh
                                st.success("✅ New API available (mediapipe.python.solutions)")
                            except:
                                st.warning("⚠️ New API not available")
                            
                            # Check old API
                            if hasattr(mp, 'solutions'):
                                st.success("✅ Legacy API available (mp.solutions)")
                            else:
                                st.warning("⚠️ Legacy API not available")
                                
                            st.info("**Solution:** Install mediapipe==0.10.13 for compatibility")
                        except ImportError as e:
                            st.error(f"❌ MediaPipe not installed: {e}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
                    
                    # Show requirements.txt content for debugging
                    with st.expander("📋 Check requirements.txt"):
                        try:
                            with open('requirements.txt', 'r') as f:
                                st.code(f.read(), language='text')
                        except:
                            st.write("Could not read requirements.txt")
                else:
                    st.info("Please ensure the model file exists at `models/hybrid_model_v2.pth`")
                return
    else:
        # Show model status if already loaded
        if st.session_state.predictor is not None:
            st.info("✅ AI Model is ready for predictions")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
        help="Upload a clear frontal face image (PNG, JPG, JPEG, GIF, or WEBP)"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, width='stretch')
        
        with col2:
            st.subheader("Prediction Results")
            
            # Predict button
            if st.button("🔮 Predict BMI", type="primary"):
                with st.spinner("Analyzing image with hybrid_model_v2.pth..."):
                    # Get image bytes
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    # Validate predictor is available
                    if st.session_state.predictor is None:
                        st.error("❌ Model predictor is not available. Please reload the page.")
                        return
                    
                    # Predict using hybrid_model_v2.pth with REAL extracted features (NO DUMMY DATA)
                    result = predict_bmi(image_bytes, st.session_state.predictor)
                    
                    if result['success']:
                        # Display results from model prediction
                        bmi = result['bmi']
                        category = result['category']
                        category_class = get_bmi_category_class(category)
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="text-align: center;">
                                <div class="bmi-value">{bmi:.2f}</div>
                                <div class="category {category_class}">{category}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show that this is from hybrid_model_v2
                        st.success("✅ Prediction completed using hybrid_model_v2.pth with real extracted features")
                        
                        # BMI Categories reference
                        st.markdown("---")
                        st.markdown("### BMI Categories Reference")
                        st.markdown("""
                        - **Underweight**: BMI < 18.5
                        - **Normal weight**: 18.5 ≤ BMI < 25
                        - **Overweight**: 25 ≤ BMI < 30
                        - **Obese**: BMI ≥ 30
                        """)
                        
                        if result.get('message'):
                            st.info(result['message'])
                    else:
                        error_msg = result.get('error', 'Prediction failed')
                        st.error(f"❌ {error_msg}")
                        
                        # Check if it's a MediaPipe error
                        if "MediaPipe" in error_msg or "mediapipe" in error_msg.lower():
                            st.warning("""
                            **MediaPipe Installation Required**
                            
                            MediaPipe is required for feature extraction. Please install it:
                            
                            ```bash
                            pip install mediapipe==0.10.13
                            ```
                            
                            Or install all requirements:
                            
                            ```bash
                            pip install -r requirements.txt
                            ```
                            
                            After installation, please restart the Streamlit app.
                            """)
                        else:
                            st.info("Please ensure the model file exists and is properly loaded.")
    else:
        st.info("👆 Please upload an image to get started")


def show_samples_page():
    """Display sample images with true BMI values"""
    st.title("Sample Images")
    st.markdown("Browse sample images with their true BMI values")
    
    # Load samples data
    try:
        csv_path = os.path.join('samples', 'dataset.csv')
        
        if not os.path.exists(csv_path):
            st.error(f"❌ Samples CSV not found at {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Validate columns
        if 'image_filename' not in df.columns or 'BMI' not in df.columns:
            st.error("❌ Required columns not found in CSV (image_filename, BMI)")
            return
        
        # Clean data
        columns_to_use = ['image_filename', 'BMI']
        if 'id' in df.columns:
            columns_to_use.insert(0, 'id')
        
        df_clean = df[columns_to_use].copy()
        df_clean = df_clean.dropna(subset=['image_filename', 'BMI'])
        df_clean['BMI'] = pd.to_numeric(df_clean['BMI'], errors='coerce')
        df_clean = df_clean.dropna(subset=['BMI'])
        
        # Check images directory
        images_dir = os.path.join('samples', 'front')
        
        if not os.path.exists(images_dir):
            st.error(f"❌ Images directory not found at {images_dir}")
            return
        
        # Get available images
        available_files = set(os.listdir(images_dir))
        available_files_lower = {f.lower(): f for f in available_files}
        
        # Filter samples with existing images
        samples_with_images = []
        for _, row in df_clean.iterrows():
            image_filename = row['image_filename']
            
            # Try exact match
            if image_filename in available_files:
                samples_with_images.append(row)
                continue
            
            # Try case-insensitive match
            if image_filename.lower() in available_files_lower:
                samples_with_images.append(row)
                continue
        
        if len(samples_with_images) == 0:
            st.warning("⚠️ No sample images found")
            return
        
        st.success(f"✅ Found {len(samples_with_images)} sample images")
        
        # Display samples in grid
        st.markdown("---")
        
        # Number of columns
        cols_per_row = 3
        
        for i in range(0, len(samples_with_images), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(samples_with_images):
                    sample = samples_with_images[idx]
                    
                    with cols[j]:
                        # Try to load and display image
                        image_filename = sample['image_filename']
                        image_path = os.path.join(images_dir, image_filename)
                        
                        # Try case-insensitive match if exact doesn't exist
                        if not os.path.exists(image_path):
                            image_filename_lower = image_filename.lower()
                            if image_filename_lower in available_files_lower:
                                image_filename = available_files_lower[image_filename_lower]
                                image_path = os.path.join(images_dir, image_filename)
                        
                        try:
                            if os.path.exists(image_path):
                                img = Image.open(image_path)
                                st.image(img, width='stretch')
                                
                                bmi = sample['BMI']
                                
                                # Determine category
                                if bmi < 18.5:
                                    category = "Underweight"
                                    category_class = "underweight"
                                elif bmi < 25:
                                    category = "Normal"
                                    category_class = "normal"
                                elif bmi < 30:
                                    category = "Overweight"
                                    category_class = "overweight"
                                else:
                                    category = "Obese"
                                    category_class = "obese"
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 0.5rem;">
                                    <div style="font-size: 1.5rem; font-weight: bold;">{bmi:.2f}</div>
                                    <div class="{category_class}" style="font-size: 1rem;">{category}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
        
    except Exception as e:
        st.error(f"❌ Error loading samples: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_privacy_policy():
    """Display privacy policy"""
    st.title("Privacy Policy & Terms")
    
    st.markdown("""
    ## Privacy Policy
    
    ### Data Collection
    This application processes images you upload to predict BMI. We want to be transparent about how we handle your data:
    
    - **Image Processing**: Images are processed in real-time and are not stored on our servers
    - **No Personal Information**: We do not collect, store, or share any personal information
    - **Temporary Processing**: Uploaded images are only held in memory during prediction and are immediately discarded
    
    ### How It Works
    1. You upload a facial image
    2. Our AI model analyzes the image
    3. The predicted BMI is returned to you
    4. The image is discarded from memory
    
    ### Disclaimer
    - This tool is for **educational and research purposes only**
    - BMI predictions from facial images are experimental and should not replace professional medical advice
    - Always consult with healthcare professionals for accurate health assessments
    - Results may vary and should not be used for medical diagnosis
    
    ### Data Security
    - All processing happens server-side with no data persistence
    - We do not use cookies or tracking technologies
    - Your privacy is our priority
    
    ### Your Rights
    - You control what images you upload
    - You can stop using the service at any time
    - No account or registration required
    
    ## Terms of Use
    
    By using this application, you agree to:
    - Use the service responsibly and legally
    - Not attempt to reverse engineer or manipulate the model
    - Understand that results are predictions, not medical diagnoses
    - Not hold the developers liable for any decisions made based on predictions
    
    ### Limitations
    - The model is trained on specific datasets and may not generalize to all populations
    - Accuracy may vary based on image quality, lighting, angle, and other factors
    - This is a research tool and should not replace professional medical assessment
    
    ## Contact
    If you have questions or concerns about this privacy policy, please contact the development team.
    
    ---
    
    **Last Updated**: January 2026
    """)

if __name__ == "__main__":
    main()
