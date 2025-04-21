import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# ----- Configuration -----
MODEL_PATH = "best_model.pth"
DATASET_PATH = "asl_alphabet_train"  # Subfolder containing all class folders

# ----- Model Definition -----
class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLModel, self).__init__()
        self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.features(x)

# ----- Load Model -----
def load_model(path):
    try:
        model = ASLModel()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")
        return None

# ----- Predict Function -----
def predict_image(image_path):
    try:
        if model is None:
            messagebox.showerror("Error", "Model not loaded. Please restart.")
            return

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        img_display = image.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_display)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

        # Load class labels from dataset folder
        classes = sorted(os.listdir(DATASET_PATH))
        predicted_class = classes[predicted.item()]
        confidence = float(probabilities[0][predicted.item()]) * 100

        result_label.configure(
            text=f"🧠 Predicted Sign: {predicted_class} ({confidence:.1f}%)",
            style="success.TLabel"
        )

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))
        result_label.configure(
            text="❌ Prediction failed",
            style="danger.TLabel"
        )

# ----- Select Image -----
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an ASL Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if file_path:
        predict_image(file_path)

# ----- Theme Switcher -----
def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    apply_theme()

def apply_theme():
    global is_dark_mode
    style.theme_use("darkly" if is_dark_mode else "cosmo")

# ----- Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", f"Model file not found: {MODEL_PATH}")
    model = None
else:
    model = load_model(MODEL_PATH)
    if model:
        model = model.to(device)
        print("Model loaded successfully")

# ----- GUI Setup -----
app = ttk.Window(title="🤟 ASL Sign Detector", themename="cosmo")
app.geometry("460x560")
app.resizable(False, False)
style = ttk.Style()
is_dark_mode = False

# ----- Title -----
title_label = ttk.Label(
    app,
    text="ASL Sign Language Detector",
    font=("Helvetica", 18, "bold"),
    style="primary.TLabel"
)
title_label.pack(pady=20)

# ----- Image Display Frame -----
image_frame = ttk.Frame(app, width=220, height=220, style="Card.TFrame")
image_frame.pack(pady=10)
image_label = ttk.Label(image_frame)
image_label.pack()

# ----- Select Image Button -----
select_button = ttk.Button(
    app,
    text="📷 Select Image",
    command=select_image,
    style="primary.TButton",
    width=20
)
select_button.pack(pady=20)

# ----- Prediction Result -----
result_label = ttk.Label(
    app,
    text="Prediction will appear here",
    font=("Helvetica", 14),
    style="info.TLabel"
)
result_label.pack(pady=10)

# ----- Theme Toggle Button -----
theme_button = ttk.Button(
    app,
    text="🌓 Toggle Theme",
    command=toggle_theme,
    style="secondary.TButton",
    width=15
)
theme_button.pack(pady=10)

# ----- Start GUI Loop -----
app.mainloop()
