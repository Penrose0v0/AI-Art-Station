import os
import cv2

from .base import BaseUpscaler
from .Real_ESRGAN.realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class RealESRGAN(BaseUpscaler):
    def __init__(
            self,
            model_path: str = "/root/share/models/Real-ESRGAN/RealESRGAN_x4plus.pth",
            scale: int = 4,
            gpu_id = None,
            half = False,
            tile = 0,
            tile_pad = 10,
            pre_pad = 0
        ):
        super().__init__()

        self.model_path = model_path
        self.scale = scale
        self.gpu_id = gpu_id
        self.half = half
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad

        self.model = self.load_model()

    def load_model(self) -> RealESRGANer:
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )
        upsampler = RealESRGANer(
            scale=self.scale,
            model_path=self.model_path,
            model=net,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half,
            gpu_id=self.gpu_id,
        )
        return upsampler

    def upscale(self, image_path: str) -> str:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        # Inference
        img_sr, _ = self.model.enhance(img, outscale=self.scale)

        # Save image
        output_dir = os.path.dirname(image_path)
        upscaled_path = os.path.join(output_dir, "upscaled.png")

        ok = cv2.imwrite(upscaled_path, img_sr)
        if not ok:
            raise IOError(f"Failed to save image: {upscaled_path}")

        return upscaled_path