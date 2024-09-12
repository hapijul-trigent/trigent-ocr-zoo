# app.py
import streamlit as st
from trocr import TROCR
from paddle_ocr import PaddleOCRProcessor
from utils import load_image, display_image
from generate_kvp import loadChain, get_kvp
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set main panel
favicon = Image.open("static/images/Trigent_Logo.png")
st.set_page_config(
    page_title="Smart Motion Insights | Trigent AXLR8 Labs",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add logo and title
logo_path = "https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png"
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{logo_path}" alt="Trigent Logo" style="max-width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)
# Main Page Title and Caption
st.title("OCR Zoo")
st.caption("Try out Various OCR models.")
st.divider()

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


# Footer with Font Awesome icons
footer_html = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<div style="text-align: center; margin-right: 10%;">
    <p>
        &copy; 2024, Trigent Software Inc. All rights reserved. |
        <a href="https://www.linkedin.com/company/trigent-software" target="_blank" aria-label="LinkedIn"><i class="fab fa-linkedin"></i></a> |
        <a href="https://www.twitter.com/trigent-software" target="_blank" aria-label="Twitter"><i class="fab fa-twitter"></i></a> |
        <a href="https://www.youtube.com/trigent-software" target="_blank" aria-label="YouTube"><i class="fab fa-youtube"></i></a>
    </p>
</div>
"""
# Custom CSS to make the footer sticky
footer_css = """
<style>
.footer {
    position: fixed;
    z-index: 1000;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
[data-testid="stSidebarNavItems"] {
    max-height: 100%!important;
}
</style>
"""
# Combining the HTML and CSS
footer = f"{footer_css}<div class='footer'>{footer_html}</div>"
# Rendering the footer
st.markdown(footer, unsafe_allow_html=True)