import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Cargar las variables de entorno del archivo .env
load_dotenv()

# Obtener las rutas de los datasets desde las variables de entorno
train_dataset_path = os.getenv("TRAIN_DATASET_PATH")
val_dataset_path = os.getenv("VAL_DATASET_PATH")

# Verificación de la disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo A: Cargar ResNet50 preentrenado
model_a = models.resnet50(pretrained=True)

# Ajustar la última capa para adaptarse a 3 clases
num_features = model_a.fc.in_features
model_a.fc = nn.Linear(num_features, 3)
model_a = model_a.to(device)

# Hiperparámetros y configuración de entrenamiento
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Transformaciones y Aumento de Datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar el Dataset de Imágenes
train_dataset = ImageFolder(root=train_dataset_path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageFolder(root=val_dataset_path, transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer_a = torch.optim.Adam(model_a.parameters(), lr=learning_rate)

# Función de entrenamiento
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# Función de validación
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Obtener predicciones
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / len(val_loader.dataset)
    return running_loss / len(val_loader), accuracy

# Loop de entrenamiento y validación para Modelo A
for epoch in range(num_epochs):
    train_loss = train(model_a, train_loader, criterion, optimizer_a, device)
    val_loss, val_accuracy = validate(model_a, val_loader, criterion, device)
    
    print(f"Modelo A - Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")

# Modelo B: Diseño propio de CNN con módulo Inception
class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        # Módulo Inception personalizado
        self.inception_1x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.inception_3x3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.inception_5x5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        # Segunda capa convolucional
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Capa totalmente conectada
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Primera capa convolucional + pooling + dropout
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Módulo inception
        x_1x1 = F.relu(self.inception_1x1(x))
        x_3x3 = F.relu(self.inception_3x3(x_1x1))
        x_5x5 = F.relu(self.inception_5x5(x_3x3))
        
        # Concatenación de los filtros de inception
        inception_output = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        
        # Segunda capa convolucional + pooling + dropout
        x = F.relu(self.conv3(inception_output))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Aplanar
        x = x.view(-1, 256 * 7 * 7)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Inicialización del modelo B
model_b = CustomCNN(num_classes=3).to(device)
optimizer_b = torch.optim.Adam(model_b.parameters(), lr=learning_rate)

# Loop de entrenamiento y validación para Modelo B
for epoch in range(num_epochs):
    train_loss = train(model_b, train_loader, criterion, optimizer_b, device)
    val_loss, val_accuracy = validate(model_b, val_loader, criterion, device)
    
    print(f"Modelo B - Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")
