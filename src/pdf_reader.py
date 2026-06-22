import fitz
from pdf2image import convert_from_path
import easyocr
import numpy as np


def extract_text_from_pdf(pdf_path):

    # Try normal PDF text extraction first
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    # If text exists, return it
    if text.strip():
        return text

    print("No text found. Using OCR...")

    # OCR fallback
    reader = easyocr.Reader(['en'])

    images = convert_from_path(pdf_path)

    ocr_text = ""

    for image in images:
       image_np = np.array(image)
       results = reader.readtext(image_np)

       for result in results:
           ocr_text += result[1] + "\n"

    return ocr_text