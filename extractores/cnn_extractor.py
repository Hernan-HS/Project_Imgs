import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocesamiento para ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Cargar ResNet50 sin capa final
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

def extraer_resnet(imagen_pil):
    """
    Extrae vector de características desde imagen PIL usando ResNet50.
    Retorna un vector de 2048 dimensiones (normalizado).
    """
    img_tensor = transform(imagen_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(img_tensor).squeeze().cpu().numpy()

    return embedding / np.linalg.norm(embedding)

