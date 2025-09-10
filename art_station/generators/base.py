from typing import Tuple
from PIL import Image
import os
import json


class BaseImageGenerator:
    
    RESOLUTION_OPTIONS = {
        "1024 × 1024 (Square)": (1024, 1024),
        "768 × 1360 (Portrait)": (768, 1360),
        "1360 × 768 (Landscape)": (1360, 768),
        "880 × 1168 (Portrait)": (880, 1168),
        "1168 × 880 (Landscape)": (1168, 880),
        "1248 × 832 (Landscape)": (1248, 832),
        "832 × 1248 (Portrait)": (832, 1248)
    }

    default_negative_prompt = (
        "low quality, cartoon, blurry, ugly, watermark, unrealistic, grainy, noisy, "
        "distorted, malformed, deformed, low resolution, extra limbs, bad anatomy, "
        "unnatural colors, jpeg artifacts, lens flare, motion blur, oversaturated, "
        "fisheye distortion, incorrect proportions, plastic texture, AI face, wrong fingers"
    )

    base_output_dir = "./outputs"

    def __init__(self):
        pass

    def load_model(self):
        pass

    def generate(
            self, 
            prompt: str,
            negative_prompt: str = None,
            seed: int = -1,
            resolution: str = None
        ) -> Tuple[Image.Image, dict]:
        pass

    def save(
            self,
            image: Image.Image,
            info: dict,
            project_name: str = None
        ) -> str:
        # Make project output dir
        proj_output_dir = os.path.join(self.base_output_dir, project_name)
        os.makedirs(proj_output_dir, exist_ok=True)

        # Make current output dir based on project dir
        index = len(os.listdir(proj_output_dir)) + 1
        output_dir = os.path.join(proj_output_dir, f"{index}")
        os.makedirs(output_dir, exist_ok=True)

        # Write info
        info["project_name"] = project_name
        with open(os.path.join(output_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        # Save image
        image_path = os.path.join(output_dir, "output.png")
        image.save(image_path)

        return image_path
