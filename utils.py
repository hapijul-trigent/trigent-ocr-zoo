from PIL import Image
import streamlit as st

def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def display_image(image):
    st.image(image, caption="Uploaded Image", use_column_width=True)
