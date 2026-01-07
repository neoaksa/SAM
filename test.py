import cv2
import torch
import numpy as np

# Your image
image = cv2.imread("bedroom_panorama.png")
print(f"🧱 Image Data (The Bricks): {image.dtype}") # Will be uint8

# The AI's internal grid
grid = torch.meshgrid(torch.linspace(0, 1, 5), torch.linspace(0, 1, 5))
print(f"📏 Internal Math (The Measuring Tape): {grid[0].dtype}") # This is usually float64 by default!