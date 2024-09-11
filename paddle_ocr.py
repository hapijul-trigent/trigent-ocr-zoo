# paddle_ocr.py
from paddleocr import PaddleOCR
from PIL import Image

class PaddleOCRProcessor:
    def __init__(self, use_gpu=False):
        # Initialize PaddleOCR with the desired configuration
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
    
    def extract_text(self, image: Image) -> str:
        # Convert PIL image to format compatible with PaddleOCR
        image_path = "temp_image.png"
        image.save(image_path)

        # Run OCR
        result = self.ocr.ocr(image_path, cls=True)
        
        # Extract and format the text
        extracted_text = ""
        for line in result:
            for word_info in line:
                # print(word_info[1][0].replace(",", " "))
                # word = word_info[1][0].replace(",", " ")
                extracted_text += word_info[1][0] + "\n"
        return extracted_text.strip()
