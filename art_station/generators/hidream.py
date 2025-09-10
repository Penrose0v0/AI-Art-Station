from typing import Tuple
from datetime import datetime

import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig
from accelerate.utils import set_seed
from PIL import Image

from .base import BaseImageGenerator
from .hi_diffusers import HiDreamImagePipeline
from .hi_diffusers import HiDreamImageTransformer2DModel
from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler


class HiDream(BaseImageGenerator):
    def __init__(
            self,
            model_type: str,
            hidream_path: str = "/root/share/models",
            llama_path: str = "/root/share/models/modelscope/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        ):
        super().__init__()

        if model_type not in ["dev", "full", "fast"]:
            raise NotImplementedError()

        # Model configurations
        self.MODEL_CONFIGS = {
            "dev": {
                "hidream_path": f"{hidream_path}/HiDream-I1-Dev",
                "llama_path": llama_path,
                "guidance_scale": 0.0,
                "num_inference_steps": 28,
                "shift": 6.0,
                "scheduler": FlashFlowMatchEulerDiscreteScheduler
            },
            "full": {
                "hidream_path": f"{hidream_path}/HiDream-I1-Full",
                "llama_path": llama_path,
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
                "shift": 3.0,
                "scheduler": FlowUniPCMultistepScheduler
            },
            "fast": {
                "hidream_path": f"{hidream_path}/HiDream-I1-Fast",
                "llama_path": llama_path,
                "guidance_scale": 0.0,
                "num_inference_steps": 16,
                "shift": 3.0,
                "scheduler": FlashFlowMatchEulerDiscreteScheduler
            }
        }

        self.config = self.MODEL_CONFIGS[model_type]
        self.pipe = self.load_model()

    def load_model(self) -> HiDreamImagePipeline:
        config = self.config
        pretrained_model_name_or_path = config["hidream_path"]
        scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
            config["llama_path"],
            use_fast=False,
        )
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            config["llama_path"],
            output_hidden_states=True,
            output_attentions=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ) #.to("cuda")
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ) #.to("cuda")
        pipe = HiDreamImagePipeline.from_pretrained(
            pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            device_map="balanced",
            torch_dtype=torch.bfloat16,
        ) #.to("cuda")
        pipe.transformer = transformer
        return pipe
    
    def generate(
            self, 
            prompt: str,
            negative_prompt: str = None,
            seed: int = -1,
            resolution: str = None
        ) -> Tuple[Image.Image, dict]:
        # Get current model configuration
        config = self.config
        guidance_scale = config["guidance_scale"]
        num_inference_steps = config["num_inference_steps"]
        
        # Parse resolution
        width, height = self.RESOLUTION_OPTIONS.get(resolution, (1360, 768))
        
        # Handle random seed
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        
        # All available GPUs should already be used by the model, no need to manually specify generator's device
        generator = torch.Generator().manual_seed(seed)
        set_seed(seed)  # Set global seed to ensure consistency of results
        
        # Execute inference
        with torch.inference_mode():
            images = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator
            ).images

        # Make info
        info = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "resolution": resolution,
            "datetime": datetime.now().isoformat()
        }
        
        return images[0], info


class HiDream_nf4(BaseImageGenerator):
    def __init__(
            self,
            # model_type: str,
            hidream_path: str = "/root/share/models",
            llama_path: str = "/root/share/models/modelscope/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        ):
        self.config = {
            "hidream_path": f"{hidream_path}/HiDream-I1-Full-nf4",
            "llama_path": llama_path
        }
        self.pipe = self.load_model()

    def load_model(self):
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(self.config["llama_path"])
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            self.config["llama_path"],
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
        )

        pipe = HiDreamImagePipeline.from_pretrained(
            "/root/share/models/HiDream-I1-Full-nf4",
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe.to("cuda")
        return pipe
    
    def generate(
            self, 
            prompt: str,
            negative_prompt: str = None,
            seed: int = -1,
            resolution: str = None
        ) -> Tuple[Image.Image, dict]:
        # Get current model configuration
        config = self.config
        guidance_scale = config["guidance_scale"]
        num_inference_steps = config["num_inference_steps"]
        
        # Parse resolution
        width, height = self.RESOLUTION_OPTIONS.get(resolution, (1360, 768))
        
        # Handle random seed
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        
        # All available GPUs should already be used by the model, no need to manually specify generator's device
        generator = torch.Generator().manual_seed(seed)
        set_seed(seed)  # Set global seed to ensure consistency of results
        
        # Execute inference
        with torch.inference_mode():
            images = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator
            ).images

        # Make info
        info = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "resolution": resolution,
            "datetime": datetime.now().isoformat()
        }
        
        return images[0], info