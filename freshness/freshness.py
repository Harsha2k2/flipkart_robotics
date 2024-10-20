import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Define a custom dataset class
class FreshnessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Define transformations for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset paths and labels from train and test folders
def load_data(folder_path):
    image_paths = []
    labels = []
    
    # Map folder names to labels (0 for fresh, 1 for rotten)
    label_map = {
        'freshapples': 0,
        'freshbanana': 0,
        'freshoranges': 0,
        'rottenapples': 1,
        'rottenbanana': 1,
        'rottenoranges': 1,
    }

    for folder_name in os.listdir(folder_path):
        if folder_name in label_map:
            folder_path_full = os.path.join(folder_path, folder_name)
            for filename in os.listdir(folder_path_full):
                if filename.endswith('.jpg') or filename.endswith('.png'):  # Include other formats if needed
                    image_paths.append(os.path.join(folder_path_full, filename))
                    labels.append(label_map[folder_name])

    return image_paths, labels

# Load training and testing data
train_image_paths, train_labels = load_data('dataset/train')
test_image_paths, test_labels = load_data('dataset/test')

# Create DataLoaders
train_dataset = FreshnessDataset(train_image_paths, train_labels, transform=transform)
test_dataset = FreshnessDataset(test_image_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define a simple CNN model using a pre-trained ResNet model
class FreshnessModel(nn.Module):
    def __init__(self):
        super(FreshnessModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Load pre-trained ResNet-18 model
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Change output layer to match classes

    def forward(self, x):
        return self.base_model(x)

# Instantiate the model and define loss function and optimizer
model = FreshnessModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress printing and loss tracking
num_epochs = 5  # Adjust number of epochs as needed
loss_values = []  # List to store loss values for plotting

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    average_loss = running_loss / len(train_loader)
    loss_values.append(average_loss)  # Store average loss for this epoch
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Save the model as an HDF5 file after training is complete
torch.save(model.state_dict(), 'fruit_freshness_model.pth')
print("Model saved as 'fruit_freshness_model.pth'")

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))  # Set x-ticks to match epochs
plt.grid()
plt.savefig('training_loss_graph.png')  # Save the plot as an image file
print("Loss graph saved as 'training_loss_graph.png'")

# Evaluation (simplified)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')