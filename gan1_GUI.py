import os
import torch
import torch.nn as nn
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import transforms

# --- GAN MODEL SETUP ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device('cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator2.pth', map_location=device))
generator.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# --- GUI APP ---
class MRIEnhancerApp:
    def __init__(self, master):
        self.master = master
        master.title("MRI Enhancer and 3D GIF Creator")

        self.input_folder = None
        self.patient_folder = None

        self.upload_button = tk.Button(master, text="Upload Patient Folder", command=self.upload_folder)
        self.upload_button.pack(pady=10)

        self.enhance_button = tk.Button(master, text="Enhance MRI Slices", command=self.enhance_mri, state=tk.DISABLED)
        self.enhance_button.pack(pady=10)

        self.gif_button = tk.Button(master, text="Create 3D GIF", command=self.create_gif, state=tk.DISABLED)
        self.gif_button.pack(pady=10)

        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

    def upload_folder(self):
        selected_folder = filedialog.askdirectory()
        if selected_folder:
            self.input_folder = selected_folder
            self.patient_folder = os.path.basename(selected_folder)
            self.enhance_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", f"Uploaded: {self.patient_folder}")

    def enhance_mri(self):
        slices_folder = self.input_folder
        output_folder = os.path.join(os.path.dirname(slices_folder), "output")
        os.makedirs(output_folder, exist_ok=True)

        slice_files = sorted([f for f in os.listdir(slices_folder) if f.endswith(('.png', '.jpg'))])

        if not slice_files:
            messagebox.showerror("Error", "No slices found in the selected folder.")
            return

        for slice_file in slice_files:
            slice_path = os.path.join(slices_folder, slice_file)
            img = Image.open(slice_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                enhanced_img_tensor = generator(img_tensor)

            enhanced_img = enhanced_img_tensor.squeeze(0).cpu().numpy()
            enhanced_img = (enhanced_img[0] * 255).clip(0, 255).astype('uint8')

            output_path = os.path.join(output_folder, slice_file)
            Image.fromarray(enhanced_img).save(output_path)

        messagebox.showinfo("Done", "MRI slices enhanced and saved!")
        self.gif_button.config(state=tk.NORMAL)

    def create_gif(self):
        output_folder = os.path.join(os.path.dirname(self.input_folder), "output")
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

        # Convert frames to 'RGB' to avoid palette-related issues when displaying GIF
        frames_rgb = [frame.convert('RGB') for frame in frames]

        frames_rgb[0].save(
            gif_output_path,
            save_all=True,
            append_images=frames_rgb[1:],
            duration=50,
            loop=0
        )

        self.gif_path = gif_output_path
        self.gif_image = Image.open(self.gif_path)
        self.gif_frames = [ImageTk.PhotoImage(frame.copy()) for frame in frames_rgb]

        self.current_frame = 0

        def update_frame():
            frame = self.gif_frames[self.current_frame]
            self.image_label.configure(image=frame)
            self.image_label.image = frame
            self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
            self.master.after(50, update_frame)

        update_frame()

        messagebox.showinfo("Done", f"GIF created and displayed!\nSaved at {gif_output_path}")


# --- MAIN ---
root = tk.Tk()
app = MRIEnhancerApp(root)
root.mainloop()
