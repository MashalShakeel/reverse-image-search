import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

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


def compute_resnet50_embedding(img: Image.Image):
    model, transform = load_model()
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()  # shape 2048
    return vec

def compute_histogram_embedding(img: Image.Image, bins=16):
    img = img.convert("RGB")
    arr = np.array(img)

    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 256))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 256))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 256))

    hist = np.concatenate([hist_r, hist_g, hist_b]).astype("float32")
    hist /= (np.linalg.norm(hist) + 1e-12)
    return hist

def compute_embedding(img: Image.Image, model_type="resnet50"):
    if model_type == "resnet50":
        return compute_resnet50_embedding(img)
    elif model_type == "hist":
        return compute_histogram_embedding(img)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
