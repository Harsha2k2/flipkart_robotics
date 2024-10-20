import spacy
import pytesseract
from PIL import Image
import pandas as pd
import os
import re
import logging
from tqdm import tqdm
from multiprocessing import Pool

# Set up logging
logging.basicConfig(filename='predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Set the path for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Lazy loading models
_spacy_model = None

def get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = spacy.load("en_core_web_sm")
    return _spacy_model

# Define the entity unit abbreviation map
entity_unit_abbreviation_map = {
    "width": {"centimetre": "cm", "foot": "ft", "millimetre": "mm", "metre": "m", "inch": "in", "yard": "yd"},
    "depth": {"centimetre": "cm", "foot": "ft", "millimetre": "mm", "metre": "m", "inch": "in", "yard": "yd"},
    "height": {"centimetre": "cm", "foot": "ft", "millimetre": "mm", "metre": "m", "inch": "in", "yard": "yd"},
    "item_weight": {"milligram": "mg", "kilogram": "kg", "microgram": "µg", "gram": "g", "ounce": "oz", "ton": "t", "pound": "lb"},
    "maximum_weight_recommendation": {"milligram": "mg", "kilogram": "kg", "microgram": "µg", "gram": "g", "ounce": "oz", "ton": "t", "pound": "lb"},
    "voltage": {"millivolt": "mV", "kilovolt": "kV", "volt": "V"},
    "wattage": {"kilowatt": "kW", "watt": "W"},
    "item_volume": {"cubic foot": "cu ft", "microlitre": "µL", "cup": "cup", "fluid ounce": "fl oz", "centilitre": "cl", "imperial gallon": "gal (imp)", "pint": "pt", "decilitre": "dl", "litre": "L", "millilitre": "ml", "quart": "qt", "cubic inch": "cu in", "gallon": "gal"}
}

# Unique units dictionary
unique_units = {
    "cm": "centimetre",
    "ft": "foot",
    "mm": "millimetre",
    "m": "metre",
    "in": "inch",
    "yd": "yard",
    "mg": "milligram",
    "kg": "kilogram",
    "µg": "microgram",
    "g": "gram",
    "oz": "ounce",
    "t": "ton",
    "lb": "pound",
    "mV": "millivolt",
    "kV": "kilovolt",
    "V": "volt",
    "kW": "kilowatt",
    "W": "watt",
    "cu ft": "cubic foot",
    "µL": "microlitre",
    "cup": "cup",
    "fl oz": "fluid ounce",
    "cl": "centilitre",
    "gal (imp)": "imperial gallon",
    "pt": "pint",
    "dl": "decilitre",
    "L": "litre",
    "ml": "millilitre",
    "qt": "quart",
    "cu in": "cubic inch",
    "gal": "gallon"
}

# Function to extract entities from an image
def extract_entities(image_path):
    try:
        # Use Tesseract to extract text from the image
        text = pytesseract.image_to_string(Image.open(image_path))
        # Process the text with spaCy
        nlp = get_spacy_model()
        doc = nlp(text)
        return doc.ents, text
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return [], ""

# Function to process a batch of rows
def process_batch(batch):
    results = []
    for row in batch:
        image_file_name = os.path.basename(row['image_link'])  # Get the filename from the URL
        image_full_path = os.path.join("/mnt/d/amazon_ml/test_image", image_file_name)  # Full path to the image
        entity_name = row["entity_name"]

        # Check if the image file exists
        if os.path.exists(image_full_path):
            entities, extracted_text = extract_entities(image_full_path)

            # Use regex to find the numeric values with units in the extracted text
            pattern = r'(\d+(\.\d+)?\s*(cm|ft|mm|m|in|yd|mg|kg|µg|g|oz|t|lb|mV|kV|V|kW|W|cu ft|µl|cup|fl oz|cl|gal (imp)|pt|dl|L|ml|qt|cu in|gal))'
            matches = re.findall(pattern, extracted_text)

            found_value = ""
            for match in matches:
                value, unit = match[0], match[2]
                for entity, units in entity_unit_abbreviation_map.items():
                    if entity == entity_name and unit in units.values():
                        found_value = f"{value} {unit}"
                        logging.info(f"Entity value found for {entity_name}: {found_value}")
                        break
                if found_value:
                    break

            results.append({"index": row.name, "prediction": found_value})
        else:
            logging.warning(f"Image not found for index {row.name}: {image_full_path}")
            results.append({"index": row.name, "prediction": ""})
    
    return results

# Function to clean and format the predictions
def format_prediction(value):
    if not value or value.strip() == "":
        return ""
    
    # Extract numeric value and unit(s)
    match = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z]+)', value)
    
    if not match:
        return ""
    
    num_val, unit = match[0]
    
    # Convert the number to the desired format (int if no decimals, else 2 decimal places)
    num_val = float(num_val)
    formatted_value = f"{int(num_val)}" if num_val.is_integer() else f"{num_val:.2f}"
    
    # Use the unique_units dictionary to map to the full unit name
    if unit in unique_units:
        unit_full = unique_units[unit]
        return f"{formatted_value} {unit_full}"
    
    return f"{formatted_value} {unit}"

# Load the test.csv file
test_df = pd.read_csv("/mnt/c/Users/harsha/Downloads/test.csv")

# Split data into batches for better multiprocessing efficiency
batch_size = 500
batches = [test_df.iloc[i:i + batch_size] for i in range(0, len(test_df), batch_size)]

# Use multiprocessing to process the batches
with Pool(processes=12) as pool:
    results = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing batches"))

# Flatten the list of results
flat_results = [item for sublist in results for item in sublist]

# Convert predictions to DataFrame
prediction_df = pd.DataFrame(flat_results)

# Format the predictions
prediction_df["prediction"] = prediction_df['prediction'].apply(format_prediction)

# Save the predictions to a CSV file
prediction_df.to_csv("predictions.csv", index=False)
logging.info("Predictions saved to predictions.csv")
