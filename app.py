import streamlit as st
from trocr import TROCR
from utils import load_image, display_image

# Initialize the TROCR instance
ocr_model = TROCR()

# Streamlit app title
st.title("TRocr Streamlit Application")

# Upload an image
uploaded_file = st.file_uploader("Upload an image for OCR", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = load_image(uploaded_file)
    if image:
        display_image(image)

        # Run OCR on the image
        with st.spinner('Extracting text from image...'):
            extracted_text = ocr_model.extract_text(image)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.text(extracted_text)
else:
    st.info("Please upload an image to extract text.")
