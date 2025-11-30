import torch
from torchvision import models

# Load ResNet architecture
model = models.resnet50(weights=None)
num_features = model.fc.in_features

# ✅ Match your training structure exactly
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(128, 3)
)

# Load weights
try:
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully! Ready for inference.")
except Exception as e:
    print("❌ Error loading model:", e)
