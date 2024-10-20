from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

# Define your model class (this should match your trained model architecture)
class FreshnessModel(nn.Module):
    def __init__(self):
        super(FreshnessModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Example: ResNet-18
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Adjust for your classes

    def forward(self, x):
        return self.base_model(x)

# Load your trained model
model = FreshnessModel()
model.load_state_dict(torch.load('fruit_freshness_model.pth'))  # Load the state dictionary
model.eval()  # Set the model to evaluation mode

# Define image transformations
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)  # Add batch dimension

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fruit Freshness Prediction API! Use the /predict endpoint to upload an image."

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            img_bytes = file.read()
            input_tensor = transform_image(img_bytes)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                class_id = predicted.item()
                class_name = "Fresh" if class_id == 0 else "Rotten"
                return jsonify({'class_id': class_id, 'class_name': class_name})
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    app.run(debug=True)