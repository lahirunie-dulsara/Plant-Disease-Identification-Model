import customtkinter as ctk
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os

# Constants
IMAGE_SIZE = 256
CLASSES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites (Two-Spotted Spider Mite)",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy",
    "Powdery Mildew",
]

# Define ResNet model
class RESNET(nn.Module):
    def __init__(self, num_classes):
        super(RESNET, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

# Load Model
model_path = r"D:\Aca\4th Sem\Rasberry Pi\trained_model_v1.pth"  # Change to your model path
model = RESNET(num_classes=len(CLASSES))
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    return CLASSES[predicted.item()], f"{max_prob.item() * 100:.2f}%"

# GUI Class
class PlantDiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Plant Disease Classifier")
        self.geometry("800x800")

        # Initialize webcam variables
        self.cap = None
        self.webcam_on = False
        self.current_image = None  # Will store either file path or PIL Image

        # Create main container
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Image display
        self.image_display = ctk.CTkLabel(
            self.main_frame, 
            text="No image selected", 
            width=600, 
            height=500
        )
        self.image_display.pack(pady=10)

        # Button frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(pady=10)

        # Upload button
        self.upload_button = ctk.CTkButton(
            self.button_frame, 
            text="Upload Image", 
            command=self.upload_image,
            width=150
        )
        self.upload_button.grid(row=0, column=0, padx=10)

        # Webcam button
        self.webcam_button = ctk.CTkButton(
            self.button_frame, 
            text="Open Webcam", 
            command=self.toggle_webcam,
            width=150
        )
        self.webcam_button.grid(row=0, column=1, padx=10)

        # Capture button (initially disabled)
        self.capture_button = ctk.CTkButton(
            self.button_frame, 
            text="Capture Image", 
            command=self.capture_image,
            width=150,
            state="disabled"
        )
        self.capture_button.grid(row=0, column=2, padx=10)

        # Predict button
        self.predict_button = ctk.CTkButton(
            self.main_frame, 
            text="Predict Disease", 
            command=self.predict_image,
            width=200
        )
        self.predict_button.pack(pady=10)

        # Prediction label
        self.prediction_label = ctk.CTkLabel(
            self.main_frame, 
            text="Prediction will appear here",
            width=600,
            height=40,
            font=("Arial", 14)
        )
        self.prediction_label.pack(pady=10)

        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def upload_image(self):
        if self.webcam_on:
            self.toggle_webcam()  # Turn off webcam if it's on
        
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.current_image = file_path  # Store file path
            self.display_image(file_path)

    def toggle_webcam(self):
        if not self.webcam_on:
            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            self.webcam_on = True
            self.webcam_button.configure(text="Close Webcam")
            self.capture_button.configure(state="normal")
            self.update_webcam_feed()
        else:
            # Close webcam
            self.webcam_on = False
            if self.cap:
                self.cap.release()
            self.webcam_button.configure(text="Open Webcam")
            self.capture_button.configure(state="disabled")
            # Don't clear the image display if we have an image

    def update_webcam_feed(self):
        if self.webcam_on:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (600, 500))
                img = Image.fromarray(frame)
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 500))
                self.image_display.configure(image=photo, text="")
                self.image_display.image = photo
                self.after(10, self.update_webcam_feed)

    def capture_image(self):
        if self.webcam_on and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(frame)  # Store PIL Image
                self.display_image(self.current_image)
                self.toggle_webcam()  # Turn off webcam after capture

    def display_image(self, image_source):
        try:
            if isinstance(image_source, str):
                # File path
                image = Image.open(image_source)
            else:
                # PIL Image
                image = image_source
            
            # Resize maintaining aspect ratio
            original_width, original_height = image.size
            ratio = min(600/original_width, 500/original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create and display image
            photo = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(600, 500)
            )
            self.image_display.configure(image=photo, text="")
            self.image_display.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def predict_image(self):
        if self.current_image is None:
            self.prediction_label.configure(
                text="No image available for prediction",
                text_color="red"
            )
            return

        try:
            # If we have a PIL Image (from webcam), save it temporarily
            if not isinstance(self.current_image, str):
                temp_path = "temp_capture.jpg"
                self.current_image.save(temp_path)
                predicted_class, confidence = predict_image(temp_path)
                os.remove(temp_path)  # Clean up
            else:
                # Regular file path
                predicted_class, confidence = predict_image(self.current_image)
            
            self.prediction_label.configure(
                text=f"Predicted: {predicted_class} with {confidence} confidence",
                text_color="white"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.destroy()

# Run the application
if __name__ == "__main__":
    app = PlantDiseaseApp()
    app.mainloop()
