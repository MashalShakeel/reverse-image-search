import torch
from torchvision import models, transforms
from PIL import Image

# loading pretrained ResNet50 once
_model = None
_transform = None

def load_model():
    global _model, _transform
    if _model is None:
        _model = models.resnet50(weights="IMAGENET1K_V2")
        _model = torch.nn.Sequential(*list(_model.children())[:-1])  # remove classifier
        _model.eval()

        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return _model, _transform


def compute_embedding(img: Image.Image):
    model, transform = load_model()
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()  # shape 2048
    return vec

