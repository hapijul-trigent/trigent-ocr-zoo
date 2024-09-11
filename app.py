# app.py
import streamlit as st
from trocr import TROCR
from paddle_ocr import PaddleOCRProcessor
from utils import load_image, display_image
from generate_kvp import loadChain, get_kvp

# Load
kvp_chain = loadChain()

# Initialize the OCR models
tocr_model = TROCR()  # TRocr
paddle_ocr_model = PaddleOCRProcessor()  # PaddleOCR

# Streamlit app title
st.title("OCR Streamlit Application")

# User selects the OCR method
ocr_option = st.selectbox("Select OCR Method:", ("TRocr", "PaddleOCR"))

# Upload an image
uploaded_file = st.file_uploader("Upload an image for OCR", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = load_image(uploaded_file)
    if image:
        display_image(image)

        # Perform OCR based on the selected option
        if ocr_option == "TRocr":
            with st.spinner('Extracting text with TRocr...'):
                extracted_text = tocr_model.extract_text(image)
        else:
            with st.spinner('Extracting text with PaddleOCR...'):
                extracted_text = paddle_ocr_model.extract_text(image)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.text(extracted_text)

        # extracted_text = {"extracted_text": extracted_text}
        df = get_kvp(extracted_text=extracted_text, llm_chain=kvp_chain)
        st.subheader('Key-Value Pair Table')
        st.dataframe(df)
else:
    st.info("Please upload an image to extract text.")
