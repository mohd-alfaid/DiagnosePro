import requests
import json  # Import JSON module
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import gdown  # Import gdown for downloading files
from streamlit_lottie import st_lottie  # Import streamlit-lottie
import tensorflow as tf  # Make sure to import tensorflow

# Function to load Lottie animation from a URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to load Lottie animation from a local JSON file
def load_lottie_local(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="DIAGNOSEPRO - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation to switch between pages
st.sidebar.title("NAVIGATION")
page = st.sidebar.selectbox("Choose a Page", ["About", "Diagnose", "Articles"])

# Function to download the model and weights from Google Drive
def download_files():
    model_url = 'https://drive.google.com/uc?id=1zpTMgXiAgvlH8c7mJy_LAn69c2eOdACb'
    weights_url = 'https://drive.google.com/uc?id=1BaCvfS5uOzZ92Idxzt9SX2-iJrro0dAt'

    # Downloading the model and weights
    gdown.download(model_url, 'my_modelvgg19.keras', quiet=False)
    gdown.download(weights_url, 'vgg_unfrozen_weights.weights.h5', quiet=False)

# Function to load the VGG model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_vgg_model():
    try:
        download_files()  # Ensure files are downloaded
        model = load_model('my_modelvgg19.keras')  
        model.load_weights('vgg_unfrozen_weights.weights.h5')  # Load weights
        st.write("Model and weights loaded successfully!")
        return model
    except Exception as e:
        st.write(f"Error loading model or weights: {e}")
        return None


# Function to make predictions with the model
def predict(image_path, model):
    try:
        # Load and preprocess the image (resize and normalize)
        img = image.load_img(image_path, target_size=(240, 240))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize image
        
        st.write(f"Image array shape: {img_array.shape}")  # Log image shape for debugging
        
        # Make predictions using the model
        predictions = model.predict(img_array)
        
        if predictions is None:
            raise ValueError("Model prediction returned None.")
        
        return predictions
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        return None

# ----------------------------------- About Page -----------------------------------
if page == "About":
    lottie_welcome = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_puciaact.json")
    lottie_health = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json")
    lottie_healthy = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json")

    # Centering the title using Markdown
    st.markdown("<h1 style='text-align: center;'>Welcome to Diagnosepro!</h1>", unsafe_allow_html=True)
    st_lottie(lottie_welcome, height=300, key="welcome")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("##")
            st.write(
                """
                This app uses deep learning to detect brain tumors from MRI images. 
                It is divided into three sections:
                - **DiagnosePro**: Upload MRI images and get real-time predictions.
                - **Articles**: Find useful information and resources about brain tumors.

                By providing high-quality predictions, we aim to support medical professionals in making accurate diagnoses and improve patient outcomes.
                """
            )
            st.write("##")
            st.write(
                "[Learn More >](https://www.example.com)"  # Replace with actual link if needed
            )
        with right_column:
            st_lottie(lottie_health, height=500, key="check")

    with st.container():
        st.write("---")
        cols = st.columns(2)
        with cols[0]:
            st.header("How it works?")
            st.write(
                """
                Our application utilizes machine learning to predict the presence of brain tumors from MRI images!
                We then recommend specialized doctors based on your type of disease. If our model predicts you're healthy, 
                we'll suggest you a general doctor.
                """
            )
        with cols[1]:
            st_lottie(lottie_healthy, height=300, key="healthy")

# ----------------------------------- DiagnosePro Page -----------------------------------
if page == "Diagnose":
    st.title("DiagnosePro - Brain Tumor Detection")

    # Load the model
    model = load_vgg_model()

    if model is None:
        st.error("Model could not be loaded. Please check the model file paths.")
    else:
        # Upload an image for tumor detection
        uploaded_file = st.file_uploader("Choose an MRI image for detection", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Save the uploaded image temporarily
                temp_file_path = os.path.join("/tmp", uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)

                st.write("Classifying the uploaded image...")

                # Make predictions using the loaded model
                predictions = predict(temp_file_path, model)

                if predictions is not None:
                    class_labels = ["No Tumor", "Tumor"]  # Assuming a binary classification: No Tumor or Tumor
                    predicted_class = np.argmax(predictions)  # Get the index of the highest prediction
                    st.write(f"The model predicts: **{class_labels[predicted_class]}**")
                else:
                    st.write("Prediction failed.")

            except Exception as e:
                st.write(f"Error during processing: {e}")

        # Add an animated sticker related to the diagnosis
        brain_animation = load_lottie_local("/Users/mohdalfaid/Desktop/brain/brain_json/brain.json")  # Make sure this path is correct
        if brain_animation:
            st_lottie(brain_animation, key="brain_lottie", height=300, width=300)
        else:
            st.write("Brain animation failed to load.")

# ----------------------------------- Articles Page -----------------------------------
elif page == "Articles":
    st.title("Learn More About Brain Tumors")

    # Load brain-related Lottie animation
    lottie_brain = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_cfnluqlz.json")
    if lottie_brain:
        st_lottie(lottie_brain, height=300, key="brain_tumor_animation")
    
    st.subheader("Explore articles related to brain tumors, diagnosis, and treatments.")

    # Article 1: Understanding Brain Tumors
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            st.header("Understanding Brain Tumors")
            st.markdown("##")
            st.markdown(
                """
                Brain tumors are abnormal growths of cells in the brain. While some brain tumors are benign, others can be cancerous. The risk factors, symptoms, and treatment options for brain tumors can vary depending on their type and location. Early diagnosis can improve outcomes.
                
                Learn more about:
                - The different types of brain tumors (benign and malignant)
                - Symptoms that may indicate a brain tumor
                - Diagnostic procedures such as MRI, CT scans, and biopsies
                - Treatment options including surgery, radiation therapy, and chemotherapy

                [Learn More](https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084#:~:text=A%20brain%20tumor%20can%20form,headaches%2C%20nausea%20and%20balance%20problems.)
                """
            )
        with cols[1]:
            st.image(
                "/Users/mohdalfaid/Desktop/brain/images/understanding.jpg",  # Placeholder image URL
                caption="What is Brain Tumor",
                use_column_width=True,
            )

    # Article 2: Brain Tumor Diagnosis and Treatment
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            st.image(
               "/Users/mohdalfaid/Desktop/brain/images/treatment.jpg",               caption="Brain Tumor Diagnosis and Treatment",
                use_column_width=True,
            )
        with cols[1]:
            st.header("Brain Tumor Diagnosis and Treatment")
            st.markdown(
                """
                Brain tumors are diagnosed through various imaging techniques and sometimes biopsies. Treatment for brain tumors typically involves a combination of surgery, radiation therapy, and chemotherapy. Other innovative treatments, such as targeted therapy, may also be available depending on the tumor type.

                For more details on brain tumor diagnosis, treatment methods, and clinical trials:
                [Learn More](https://www.mayoclinic.org/diseases-conditions/brain-tumor/diagnosis-treatment/drc-20350088)
                """
            )

    # Article 3: Advances in Brain Tumor Research
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            st.header("Advances in Brain Tumor Research")
            st.markdown(
                """
                Ongoing research in brain tumor treatment has led to innovative therapies that target specific cancer cells. Clinical trials are testing new drugs, gene therapies, and immunotherapy approaches to treat brain tumors more effectively.

                Explore research updates:
                - Ongoing clinical trials for brain tumors
                - Emerging therapies and technologies in the treatment of brain tumors
                - The role of genetics in brain tumors and personalized medicine

                [Learn More](https://www.cancer.gov/types/brain/research)
                """
            )
        with cols[1]:
            st.image(
            "/Users/mohdalfaid/Desktop/brain/images/Advances.jpg",              caption="Advancements in  Brain Tumor Treatments",
                use_column_width=True,
            )

    # Article 4: Living with a Brain Tumor
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            st.image(
                "/Users/mohdalfaid/Desktop/brain/images/living.jpg",
                caption="Living with a Brain Tumor",
                use_column_width=True,
            )
        with cols[1]:
            st.header("Living with a Brain Tumor")
            st.markdown(
                """
                A brain tumor diagnosis can significantly impact a person's quality of life. However, with the right treatment and support, many individuals continue to lead fulfilling lives. Rehabilitation, counseling, and support groups can help patients cope with the emotional and physical challenges of living with a brain tumor.

                Explore resources for:
                - Rehabilitation and physical therapy
                - Emotional and mental health support
                - Caregiver support and resources

                [Learn More](https://www.mountelizabeth.com.sg/conditions-diseases/brain-tumours/symptoms-causes?gclid=CjwKCAjw9p24BhB_EiwA8ID5BopmiYzzcKZd0l7WfVhwHycqHk3Dt0CWT3Y3gG4Dy0KVg9SB5M_rHRoCV5AQAvD_BwE)
                """
            )

    # Article 5: Pediatric Brain Tumors
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            st.header("Pediatric Brain Tumors")
            st.markdown(
                """
                Brain tumors are the most common solid tumors in children. While many childhood brain tumors are treatable, they require specialized care. Understanding the unique challenges faced by pediatric brain tumor patients and their families is essential for providing the best possible treatment.

                Learn more about:
                - Types of brain tumors commonly found in children
                - Treatment protocols for pediatric brain tumors
                - The long-term effects of treatment on children's development and quality of life

                [Learn More](https://www.mayoclinic.org/diseases-conditions/pediatric-brain-tumor/symptoms-causes/syc-20361694)
                """
            )
        with cols[1]:
            st.image(
                "/Users/mohdalfaid/Desktop/brain/images/Pediatric.jpg",
                caption="Pediatric Brain Tumors",
                use_column_width=True,
            )

    st.write("---")
    st.write("For more resources, visit the links provided above or consult your doctor.")
