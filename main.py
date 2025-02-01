import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)

image_path = "images/content.jpg"
try:
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.title("Image Loaded Successfully")
    plt.show()
except Exception as e:
    print("Error loading image:", e)