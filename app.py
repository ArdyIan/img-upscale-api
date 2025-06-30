# app.py
import gradio as gr
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import os

# Fungsi upscaling Anda yang sudah ada
def upscale_image_gradio(input_image):
    # Pastikan direktori models ada
    model_path = 'models/RealESRGAN_x4plus.pth'
    if not os.path.exists(model_path):
        # Anda perlu memastikan model .pth diunduh ke folder 'models'
        # atau Anda bisa mengunduhnya secara programatik jika tidak ada
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download it.")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0, # Atur tile sesuai kebutuhan memori GPU/CPU Anda
        tile_pad=10,
        pre_pad=0,
        half=False # Atur True jika Anda menggunakan GPU yang mendukung half-precision (FP16)
    )

    # input_image dari Gradio biasanya adalah numpy array (dari PIL Image yang di-load)
    # Pastikan formatnya sesuai (BGR untuk OpenCV jika model Anda dilatih dengan itu)
    # Gradio mengembalikan RGB, Real-ESRGAN/OpenCV biasanya butuh BGR
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    output_bgr, _ = upsampler.enhance(img_bgr, outscale=4)

    # Konversi kembali ke RGB untuk ditampilkan oleh Gradio
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    
    return output_rgb

# Definisikan antarmuka Gradio
iface = gr.Interface(
    fn=upscale_image_gradio,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Upscaled Image"),
    title="Real-ESRGAN Image Upscaler",
    description="Upload an image to upscale it using Real-ESRGAN x4plus model."
)

if __name__ == "__main__":
    iface.launch() # Untuk menjalankan lokal