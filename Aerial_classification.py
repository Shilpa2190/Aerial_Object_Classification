import torch
from PIL import Image
import torch.nn as nn
import streamlit as st
from torchvision import transforms, models
from streamlit_option_menu import option_menu
import numpy as np


st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")

with st.sidebar:
    selected = option_menu("Main Menu", ["Classification"]) 
        
if selected == "Classification":
                st.title("Bird vs Drone Image Classifier")
                st.write("Upload an image and the model to classify it.")



                # --- Upload model ---
                model_file = st.file_uploader("Upload your trained model (.pth file)", type=["pth"])
                if model_file:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    num_classes = 2

                    # Load pretrained ResNet model
                    model = models.resnet18(pretrained=False)
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, num_classes)
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.to(device)
                    model.eval()

                    st.success("Model loaded successfully!")

                # --- Upload image ---
                image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if image_file and model_file:
                    image = Image.open(image_file).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    # --- Image transforms ---
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(image).unsqueeze(0).to(device)

                    # --- Prediction ---
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, pred = torch.max(outputs, 1)

                    classes = ["bird", "drone"]
                    st.write(f"Predicted Class: **{classes[pred.item()]}**")

