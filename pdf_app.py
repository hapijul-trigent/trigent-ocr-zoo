import streamlit as st
from trocr import TROCR
from paddle_ocr import PaddleOCRProcessor
from PIL import ImageEnhance, Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from utils import load_image, display_image

# Initialize the OCR models
tocr_model = TROCR()  # TRocr
paddle_ocr_model = PaddleOCRProcessor()  # PaddleOCR

# Streamlit app title
st.title("OCR Streamlit Application")

# User selects the OCR method
ocr_option = st.selectbox("Select OCR Method:", ("TRocr", "PaddleOCR"))

# File uploader to accept both images and PDFs
uploaded_file = st.file_uploader("Upload an image or PDF for OCR", type=["jpg", "png", "jpeg", "pdf"])

# Preprocess function (previously from utils.py)
def preprocess_image(image):
    # Convert to RGB and enhance contrast
    image = image.convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast (can be adjusted)
    return image

# Function to process PDF and extract text from images
def process_pdf(file, ocr_model, ocr_option):
    images = convert_from_path(file)  # Convert each PDF page to an image
    extracted_text = ""
    
    for i, image in enumerate(images):
        # Preprocess each page image
        image = preprocess_image(image)
        display_image(image)
        
        # Extract text using the selected OCR method
        if ocr_option == "TRocr":
            with st.spinner(f'Extracting text from page {i+1} with TRocr...'):
                text = ocr_model.extract_text(image)
        else:
            with st.spinner(f'Extracting text from page {i+1} with PaddleOCR...'):
                text = paddle_ocr_model.extract_text(image)
        
        extracted_text += f"Page {i+1}:\n{text}\n\n"
    
    return extracted_text

# Function to extract text directly from PDF (embedded text)
def extract_text_from_pdf(file):
    pdf_document = fitz.open(file)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += f"Page {page_num+1}:\n{page.get_text()}\n\n"
    return text

if uploaded_file is not None:
    # Check if the uploaded file is an image or a PDF
    if uploaded_file.type == "application/pdf":
        # Let the user decide whether to extract embedded text or run OCR on PDF pages
        extract_method = st.radio("Select PDF text extraction method:", ("Embedded Text", "OCR on PDF Pages"))
        
        if extract_method == "Embedded Text":
            with st.spinner('Extracting embedded text from PDF...'):
                extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            # Process PDF and run OCR on each page image
            extracted_text = process_pdf(uploaded_file, tocr_model, ocr_option)
    else:
        # If the file is an image, process it normally
        image = load_image(uploaded_file)
        if image:
            # Preprocess the image
            image = preprocess_image(image)
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
else:
    st.info("Please upload an image or PDF to extract text.")
