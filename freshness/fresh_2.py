import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define the model class (same as used during training)
class FreshnessModel(nn.Module):
    def __init__(self):
        super(FreshnessModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Load pre-trained ResNet-18 model
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Change output layer to match classes

    def forward(self, x):
        return self.base_model(x)

# Load the trained model weights
model = FreshnessModel()
model_weights_path = 'fruit_freshness_model.pth'

# Use weights_only=True to avoid security warnings
model.load_state_dict(torch.load(model_weights_path, weights_only=True))
model.eval()  # Set to evaluation mode

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict freshness from a single image
def predict_freshness(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(img)  # Forward pass through the model
        _, predicted = torch.max(output.data, 1)  # Get the predicted class
    
    # Map predicted class to freshness status
    freshness_status = {
        0: "Fresh",
        1: "Rotten"
    }

    return f'The fruit is: {freshness_status[predicted.item()]}'

# Example usage
image_path = 'C:/Users/harsha/Downloads/xyzz.jpg'  # Replace with your image path
result = predict_freshness(image_path)
print(result)