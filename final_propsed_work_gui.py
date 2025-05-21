import os
import threading
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import transforms

device = torch.device('cpu')

# --- NESRGAN MODEL SETUP ---
class NESRGAN_ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(NESRGAN_ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class NESRGAN_Generator(nn.Module):
    def __init__(self):
        super(NESRGAN_Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential(
            NESRGAN_ResidualBlock(64),
            NESRGAN_ResidualBlock(64),
            NESRGAN_ResidualBlock(64)
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

# --- GUI ---
class MRIEnhancerApp:
    def __init__(self, master):
        self.master = master
        master.title("MRI Enhancer and 3D GIF Creator")

        self.input_folder = None
        self.patient_folder = None
        self.gif_frames = []
        self.current_frame = 0
        self.is_paused = False

        self.upload_button = tk.Button(master, text="Upload Patient Folder", command=self.upload_folder)
        self.upload_button.pack(pady=10)

        self.nesrgan_button = tk.Button(master, text="Apply NESRGAN", command=self.start_nesrgan, state=tk.DISABLED)
        self.nesrgan_button.pack(pady=10)

        self.gif_button = tk.Button(master, text="Create 3D GIF", command=self.create_gif, state=tk.DISABLED)
        self.gif_button.pack(pady=10)

        self.pause_button = tk.Button(master, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

    def upload_folder(self):
        selected_folder = filedialog.askdirectory()
        if selected_folder:
            self.input_folder = selected_folder
            self.patient_folder = os.path.basename(selected_folder)
            self.nesrgan_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", f"Uploaded: {self.patient_folder}")

    def start_nesrgan(self):
        self.disable_buttons()
        threading.Thread(target=self.apply_nesrgan).start()

    def disable_buttons(self):
        self.nesrgan_button.config(state=tk.DISABLED)
        self.gif_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.nesrgan_button.config(state=tk.NORMAL)
        self.gif_button.config(state=tk.NORMAL)

    def apply_nesrgan(self):
        self.enhance_mri(NESRGAN_Generator, 'nesrgan_generator_finetuned.pth', "NESRGAN")

    def enhance_mri(self, generator_class, model_weights, generator_name):
        try:
            output_folder = os.path.join(os.path.dirname(self.input_folder), f"enhanced_output")
            if os.path.exists(output_folder):
                messagebox.showinfo("Skipped", f"Enhancement already done: {generator_name}")
                self.master.after(0, self.enable_buttons)
                return

            generator = generator_class().to(device)
            generator.load_state_dict(torch.load(model_weights, map_location=device))
            generator.eval()

            os.makedirs(output_folder, exist_ok=True)

            for filename in os.listdir(self.input_folder):
                if filename.endswith(".png"):
                    img_path = os.path.join(self.input_folder, filename)
                    img = Image.open(img_path).convert('L')
                    original_size = img.size

                    transform = transforms.Compose([transforms.ToTensor()])
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        enhanced_tensor = generator(img_tensor)

                    enhanced_img = enhanced_tensor.squeeze(0).cpu().numpy()
                    enhanced_img = (enhanced_img[0] * 255).clip(0, 255).astype(np.uint8)
                    enhanced_pil = Image.fromarray(enhanced_img).resize(original_size, Image.BICUBIC)

                    output_image_path = os.path.join(output_folder, f"{generator_name}_{filename}")
                    enhanced_pil.save(output_image_path)

            self.master.after(0, self.enable_buttons)
            messagebox.showinfo("Done", f"Enhancement done: {generator_name}")
        except Exception as e:
            self.master.after(0, self.enable_buttons)
            messagebox.showerror("Error", f"Enhancement failed: {str(e)}")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")

    def create_gif(self):
        try:
            output_folder = os.path.join(os.path.dirname(self.input_folder), f"enhanced_output")
            gif_output_path = os.path.join(os.path.dirname(self.input_folder), f"{self.patient_folder}_enhanced.gif")

            slice_files = sorted([f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg'))])
            frames = []

            for slice_file in slice_files:
                slice_path = os.path.join(output_folder, slice_file)
                img = Image.open(slice_path).convert('L')
                frames.append(img)

            if not frames:
                messagebox.showerror("Error", "No enhanced slices found.")
                return

            frames_rgb = [frame.convert('RGB') for frame in frames]

            frames_rgb[0].save(
                gif_output_path,
                save_all=True,
                append_images=frames_rgb[1:],
                duration=50,
                loop=0
            )

            self.gif_frames = [ImageTk.PhotoImage(frame.copy()) for frame in frames_rgb]
            self.current_frame = 0
            self.pause_button.config(state=tk.NORMAL)

            def update_frame():
                if self.gif_frames:
                    if not self.is_paused:
                        frame = self.gif_frames[self.current_frame]
                        self.image_label.configure(image=frame)
                        self.image_label.image = frame
                        self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
                    self.master.after(50, update_frame)

            update_frame()

            messagebox.showinfo("Done", f"GIF created and displayed!\nSaved at {gif_output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"GIF creation failed: {str(e)}")

# --- MAIN ---
root = tk.Tk()
app = MRIEnhancerApp(root)
root.mainloop()
