import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

def upscale_image(input_path, output_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='models/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    output, _ = upsampler.enhance(img, outscale=4)
    cv2.imwrite(output_path, output)
    return output_path
