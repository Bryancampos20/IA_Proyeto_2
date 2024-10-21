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
from torch.cuda.amp import GradScaler
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Usar el backend sin interfaz gráfica
matplotlib.use('Agg')

# Cargar las variables de entorno del archivo .env
load_dotenv()

# Inicialización de WandB
wandb.init(
    project="model_comparison",
    name="comparison_run_modelA_modelB",
    config={
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 10,
        "architecture_A": "ResNet50",
        "architecture_B": "CustomCNN",
        "dataset": "ImageFolder"
    }
)

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
batch_size = 64
learning_rate = 0.001
num_epochs = 10
scaler = GradScaler()

# Transformaciones con Data Augmentation para el conjunto de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformación básica sin Data Augmentation para validación
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
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

# Definir la función de pérdida
criterion = nn.CrossEntropyLoss()

# Función de entrenamiento con WandB y cálculo de precisión
def train(model, train_loader, criterion, optimizer, device, model_name, dataset_name, epoch):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        total_samples += labels.size(0)  # Contar muestras para calcular accuracy

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Calcular el número de predicciones correctas
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    average_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    # Registrar loss y accuracy en WandB
    wandb.log({
        f"{model_name}_{dataset_name}_train_loss": average_loss,
        f"{model_name}_{dataset_name}_train_accuracy": accuracy,
        'epoch': epoch
    })

    return average_loss, accuracy

# Función de validación con WandB y matriz de confusión
def validate_and_plot_confusion_matrix(model, val_loader, criterion, device, model_name, dataset_name, epoch):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    accuracy = correct_predictions / len(val_loader.dataset)
    average_loss = running_loss / len(val_loader)

    wandb.log({
        f"{model_name}_{dataset_name}_val_loss": average_loss,
        f"{model_name}_{dataset_name}_val_accuracy": accuracy,
        'epoch': epoch
    })

    # Generar la matriz de confusión
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Crear la estructura de carpetas para las matrices de confusión
    output_dir = os.path.join("confusion_matrices", model_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Guardar la matriz de confusión en la carpeta correspondiente
    file_path = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch}.png')
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión - {model_name} - {dataset_name} - Epoch {epoch}')
    plt.savefig(file_path)
    plt.close()

    return average_loss, accuracy

# Definir la clase CustomCNN antes de usar model_b
class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()

        # Primera capa convolucional con stride aumentado para reducir el tamaño
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # Módulo Inception personalizado
        self.inception_1x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.inception_3x3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.inception_5x5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(96)

        # Segunda capa convolucional con stride aumentado
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.4)

        # Capas totalmente conectadas
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # Ajuste del tamaño correcto
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Primera capa convolucional + pooling + dropout
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Módulo inception
        x_1x1 = F.relu(self.inception_1x1(x))
        x_3x3 = F.relu(self.inception_3x3(x_1x1))
        x_5x5 = F.relu(self.inception_5x5(x_1x1))

        inception_output = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        x = self.batch_norm2(inception_output)

        # Segunda capa convolucional + pooling + dropout
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.batch_norm3(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Aplanar el tensor antes de pasarlo a la capa totalmente conectada
        x = x.view(x.size(0), -1)

        # Capas completamente conectadas
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

# Inicialización del modelo B y su optimizador
model_b = CustomCNN(num_classes=3).to(device)
optimizer_b = torch.optim.Adam(model_b.parameters(), lr=learning_rate)

# Loop de entrenamiento y validación para Modelo A y B en los tres conjuntos de datos
optimizer_a = torch.optim.Adam(model_a.parameters(), lr=learning_rate)
for epoch in range(1, num_epochs + 1):
    print(f"\n--- Epoch [{epoch}/{num_epochs}] ---")

    # Entrenamiento y validación en dataset crudo para Modelo A
    train_loss = train(model_a, train_raw_loader, criterion, optimizer_a, device, "Model_A", "Raw_Data", epoch)
    val_loss, val_accuracy = validate_and_plot_confusion_matrix(model_a, val_loader, criterion, device, "Model_A", "Raw_Data", epoch)
    print(f"Modelo A - Raw Data - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset bilateral para Modelo A
    train_loss = train(model_a, train_bilateral_loader, criterion, optimizer_a, device, "Model_A", "Bilateral_Filter", epoch)
    val_loss, val_accuracy = validate_and_plot_confusion_matrix(model_a, val_loader, criterion, device, "Model_A", "Bilateral_Filter", epoch)
    print(f"Modelo A - Bilateral Filter - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset canny para Modelo A
    train_loss = train(model_a, train_canny_loader, criterion, optimizer_a, device, "Model_A", "Canny_Edge", epoch)
    val_loss, val_accuracy = validate_and_plot_confusion_matrix(model_a, val_loader, criterion, device, "Model_A", "Canny_Edge", epoch)
    print(f"Modelo A - Canny Edge - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset crudo para Modelo B
    train_loss = train(model_b, train_raw_loader, criterion, optimizer_b, device, "Model_B", "Raw_Data", epoch)
    val_loss, val_accuracy = validate_and_plot_confusion_matrix(model_b, val_loader, criterion, device, "Model_B", "Raw_Data", epoch)
    print(f"Modelo B - Raw Data - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset bilateral para Modelo B
    train_loss = train(model_b, train_bilateral_loader, criterion, optimizer_b, device, "Model_B", "Bilateral_Filter", epoch)
    val_loss, val_accuracy = validate_and_plot_confusion_matrix(model_b, val_loader, criterion, device, "Model_B", "Bilateral_Filter", epoch)
    print(f"Modelo B - Bilateral Filter - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Entrenamiento y validación en dataset canny para Modelo B
    train_loss = train(model_b, train_canny_loader, criterion, optimizer_b, device, "Model_B", "Canny_Edge", epoch)
    val_loss, val_accuracy = validate_and_plot_confusion_matrix(model_b, val_loader, criterion, device, "Model_B", "Canny_Edge", epoch)
    print(f"Modelo B - Canny Edge - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
