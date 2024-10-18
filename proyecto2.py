import os
import cv2
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Cargar las variables de entorno del archivo .env
load_dotenv()

# Obtener las rutas de los datasets desde las variables de entorno
train_dataset_path = os.getenv("TRAIN_DATASET_PATH")
val_dataset_path = os.getenv("VAL_DATASET_PATH")

# Verificación de la disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo A: Cargar ResNet50 preentrenado
model_a = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_features = model_a.fc.in_features
model_a.fc = nn.Linear(num_features, 3)
model_a = model_a.to(device)

# Hiperparámetros y configuración de entrenamiento
batch_size = 4
learning_rate = 0.001
num_epochs = 2
scaler = GradScaler()

# Transformaciones con Data Augmentation para el conjunto de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformación básica sin Data Augmentation para validación
val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Funciones de preprocesamiento de imágenes manteniendo la estructura de clases
def copy_images_with_class_structure(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, _ in os.walk(input_dir):
        for class_folder in dirs:
            class_input_path = os.path.join(input_dir, class_folder)
            class_output_path = os.path.join(output_dir, class_folder)
            os.makedirs(class_output_path, exist_ok=True)

            for file in os.listdir(class_input_path):
                if file.endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_input_path, file)
                    save_path = os.path.join(class_output_path, file)
                    cv2.imwrite(save_path, cv2.imread(img_path))

def apply_bilateral_filter(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, _ in os.walk(input_dir):
        for class_folder in dirs:
            class_input_path = os.path.join(input_dir, class_folder)
            class_output_path = os.path.join(output_dir, class_folder)
            os.makedirs(class_output_path, exist_ok=True)

            for file in os.listdir(class_input_path):
                if file.endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_input_path, file)
                    img = cv2.imread(img_path)
                    filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
                    save_path = os.path.join(class_output_path, file)
                    cv2.imwrite(save_path, filtered_img)

def apply_canny_filter(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, _ in os.walk(input_dir):
        for class_folder in dirs:
            class_input_path = os.path.join(input_dir, class_folder)
            class_output_path = os.path.join(output_dir, class_folder)
            os.makedirs(class_output_path, exist_ok=True)

            for file in os.listdir(class_input_path):
                if file.endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_input_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    filtered_img = cv2.Canny(img, 100, 200)
                    save_path = os.path.join(class_output_path, file)
                    cv2.imwrite(save_path, filtered_img)

# Crear datasets preprocesados si no existen
processed_dir = "processed_datasets"
raw_data_dir = os.path.join(processed_dir, "raw")
bilateral_data_dir = os.path.join(processed_dir, "bilateral")
canny_data_dir = os.path.join(processed_dir, "canny")

if not os.path.exists(raw_data_dir):
    copy_images_with_class_structure(train_dataset_path, raw_data_dir)

if not os.path.exists(bilateral_data_dir):
    apply_bilateral_filter(train_dataset_path, bilateral_data_dir)

if not os.path.exists(canny_data_dir):
    apply_canny_filter(train_dataset_path, canny_data_dir)

# Crear los loaders de los tres conjuntos de datos
train_raw_dataset = ImageFolder(root=raw_data_dir, transform=train_transform)
train_bilateral_dataset = ImageFolder(root=bilateral_data_dir, transform=train_transform)
train_canny_dataset = ImageFolder(root=canny_data_dir, transform=train_transform)

train_raw_loader = DataLoader(dataset=train_raw_dataset, batch_size=batch_size, shuffle=True)
train_bilateral_loader = DataLoader(dataset=train_bilateral_dataset, batch_size=batch_size, shuffle=True)
train_canny_loader = DataLoader(dataset=train_canny_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageFolder(root=val_dataset_path, transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()

# Función de entrenamiento con precisión mixta
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / len(val_loader.dataset)
    return running_loss / len(val_loader), accuracy

# Loop de entrenamiento y validación para Modelo A en los tres conjuntos de datos
optimizer_a = torch.optim.Adam(model_a.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    print(f"\n--- Epoch [{epoch+1}/{num_epochs}] ---")
    
    # Entrenamiento y validación en dataset crudo
    train_loss = train(model_a, train_raw_loader, criterion, optimizer_a, device)
    val_loss, val_accuracy = validate(model_a, val_loader, criterion, device)
    print(f"Modelo A - Raw Data - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset bilateral
    train_loss = train(model_a, train_bilateral_loader, criterion, optimizer_a, device)
    val_loss, val_accuracy = validate(model_a, val_loader, criterion, device)
    print(f"Modelo A - Bilateral Filter - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset canny
    train_loss = train(model_a, train_canny_loader, criterion, optimizer_a, device)
    val_loss, val_accuracy = validate(model_a, val_loader, criterion, device)
    print(f"Modelo A - Canny Edge - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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
        self.inception_3x3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.inception_5x5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        # Segunda capa convolucional
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Capas totalmente conectadas
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes
        
    def forward(self, x):
        # Primera capa convolucional + pooling + dropout
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Módulo inception
        x_1x1 = F.relu(self.inception_1x1(x))
        x_3x3 = F.relu(self.inception_3x3(x_1x1))
        x_5x5 = F.relu(self.inception_5x5(x_1x1))
        
        # Concatenación de los filtros de inception
        inception_output = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        
        # Segunda capa convolucional + pooling + dropout
        x = F.relu(self.conv3(inception_output))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Aplanar y pasar por capas totalmente conectadas
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512).to(x.device)
            self.fc2 = nn.Linear(512, self.num_classes).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Inicialización del modelo B
model_b = CustomCNN(num_classes=3).to(device)
optimizer_b = torch.optim.Adam(model_b.parameters(), lr=learning_rate)

# Loop de entrenamiento y validación para Modelo B en los tres conjuntos de datos
for epoch in range(num_epochs):
    print(f"\n--- Epoch [{epoch+1}/{num_epochs}] ---")
    
    # Entrenamiento y validación en dataset crudo
    train_loss = train(model_b, train_raw_loader, criterion, optimizer_b, device)
    val_loss, val_accuracy = validate(model_b, val_loader, criterion, device)
    print(f"Modelo B - Raw Data - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset bilateral
    train_loss = train(model_b, train_bilateral_loader, criterion, optimizer_b, device)
    val_loss, val_accuracy = validate(model_b, val_loader, criterion, device)
    print(f"Modelo B - Bilateral Filter - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset canny
    train_loss = train(model_b, train_canny_loader, criterion, optimizer_b, device)
    val_loss, val_accuracy = validate(model_b, val_loader, criterion, device)
    print(f"Modelo B - Canny Edge - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
