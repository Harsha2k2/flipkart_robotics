from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Function for Brand Recognition using TrOCR
def recognize_brand(image_path):
    # Load the pretrained model and processor for TrOCR directly from Hugging Face
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB").resize((384, 384))  # Resize to expected input size

    # Process the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    # Decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

# Function for Object Counting using ResNet
def count_objects(image_path):
    # Load a pretrained ResNet model directly from Hugging Face
    model = resnet50(weights='DEFAULT')  # Use weights instead of pretrained
    model.eval()

    # Define transformation for input images (ensure RGB)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ResNet
    ])

    # Load and preprocess the image (convert to RGB)
    image = Image.open(image_path).convert('RGB')  # Ensure it is RGB now
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Post-process output to get count (this will depend on your specific model)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 categories and their probabilities (modify as needed)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    return top5_catid, top5_prob

# Main function to run both tasks
def main(image_path):
    print("Recognizing Brand...")
    brand_text = recognize_brand(image_path)
    print("Recognized Brand Text:", brand_text)

    print("\nCounting Objects...")
    top_categories, probabilities = count_objects(image_path)

    print("Top 5 Predictions:")
    for i in range(top_categories.size(0)):
        print(f"{i + 1}: Category ID {top_categories[i].item()} - Probability: {probabilities[i].item():.4f}")

# Replace with your image path
image_path = 'C:/Users/harsha/Downloads/boost.jpg'  # Update this path

if __name__ == "__main__":
    main(image_path)