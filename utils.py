# utils.py
from PIL import Image, ImageEnhance
import streamlit as st
from trocr import TROCR
from paddle_ocr import PaddleOCRProcessor

def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB to ensure compatibility
        image = image.convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def preprocess_image(image):
    # Convert to grayscale and enhance contrast
    image = image.convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    return image

def display_image(image):
    st.image(image, caption="Uploaded Image", use_column_width=True)

st.cache_resource(show_spinner=False)
def load_trOCR():
    return TROCR()  # TRocr

st.cache_resource(show_spinner=False)
def load_PaddleOCR():
    return PaddleOCRProcessor()