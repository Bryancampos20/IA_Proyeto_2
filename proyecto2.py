import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Cargar las variables de entorno del archivo .env
load_dotenv()

# Obtener las rutas de los datasets desde las variables de entorno
train_dataset_path = os.getenv("TRAIN_DATASET_PATH")
val_dataset_path = os.getenv("VAL_DATASET_PATH")

# Verificación de la disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar ResNet50 preentrenado
model = models.resnet50(pretrained=True)

# Ajustar la última capa para adaptarse a 3 clases
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
model = model.to(device)

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

# Loop de entrenamiento y validación
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")
