import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Set device
device = torch.device('cpu')

# Define Residual Block (Missing in your code)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Define Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.conv_mid(x)
        x = x + initial
        x = self.upsample(x)
        return x

# Instantiate the Generator
generator = Generator().to(device)

# Load pre-trained generator weights
generator.load_state_dict(torch.load('nesrgan_generator_finetuned.pth', map_location=device))

# Set model to evaluation mode
generator.eval()

# Load the MRI image
image_path = 'enhanced_mri.png'
img = Image.open(image_path).convert('L')  # Grayscale

# Preprocess image
transform = transforms.Compose([
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# Pass through generator
with torch.no_grad():
    enhanced_img_tensor = generator(img_tensor)

# Postprocess output
enhanced_img = enhanced_img_tensor.squeeze(0).cpu().numpy()  # [C, H, W]
enhanced_img = (enhanced_img[0] * 255).clip(0, 255).astype(np.uint8)  # Take only 1 channel

# Save the enhanced MRI image
output_folder = 'enhanced_images'
os.makedirs(output_folder, exist_ok=True)
output_image_path = os.path.join(output_folder, 'enhanced_mri.png')
Image.fromarray(enhanced_img).save(output_image_path)

# Show original and enhanced images
plt.subplot(1, 2, 1)
plt.title('Original MRI')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Enhanced MRI')
plt.imshow(enhanced_img, cmap='gray')
plt.axis('off')

plt.show()

print(f"Enhanced image saved at: {output_image_path}")
