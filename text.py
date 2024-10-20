import pytesseract
from PIL import Image
import pandas as pd
import re
import logging

# Set up logging
logging.basicConfig(filename='ocr_predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Set the path for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

# Function to extract text from an image
def extract_text(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return ""

# Function to extract details using regex
def extract_details(text):
    brand_name = re.search(r'Brand:\s*(.*)', text)  # Adjust regex based on actual text format
    pack_size = re.search(r'Pack Size:\s*(.*)', text)  # Adjust regex based on actual text format
    expiry_date = re.search(r'Expiry Date:\s*(.*)', text)  # Adjust regex based on actual text format
    mrp = re.search(r'MRP:\s*(.*)', text)  # Adjust regex based on actual text format

    return {
        'Brand Name': brand_name.group(1) if brand_name else None,
        'Pack Size': pack_size.group(1) if pack_size else None,
        'Expiry Date': expiry_date.group(1) if expiry_date else None,
        'MRP': mrp.group(1) if mrp else None,
    }

# Specify the path to your image
image_path = r"C:/Users/harsha/Downloads/boost.jpg"  # Update this path

# Extract text from the specified image
extracted_text = extract_text(image_path)

# Extract details from the text
details = extract_details(extracted_text)

# Print extracted details in terminal
print("Extracted Details:")
for key, value in details.items():
    print(f"{key}: {value}")

# Save details to a CSV file
df = pd.DataFrame([details])  # Create DataFrame from the details dictionary
df.to_csv("extracted_details.csv", index=False)
logging.info("Extracted details saved to extracted_details.csv")