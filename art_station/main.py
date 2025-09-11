import gradio as gr

from generators.hidream import HiDream
from generators.base import BaseImageGenerator

from upscalers.real_esrgan import RealESRGAN
from upscalers.base import BaseUpscaler


def make_image_generator() -> BaseImageGenerator:
    # Todo: variable models
    return HiDream(model_type="full")

def make_upscaler() -> BaseUpscaler:
    # Todo: variable models
    return RealESRGAN()


class ArtStation:
    def __init__(self):
        self.image_generator = make_image_generator()
        self.upscaler = make_upscaler()

    def _gen_and_save(
            self, 
            prompt: str,
            negative_prompt: str = None,
            seed: int = -1,
            resolution: str = None,
            project_name: str = None
        ) -> str:
        # Check input
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Prompt cannot be empty")
        
        negative = None if negative_prompt is None or str(negative_prompt).strip() == "" \
              else negative_prompt
        
        try:
            seed = int(seed)
        except Exception:
            seed = -1

        # Generate
        image, info = self.image_generator.generate(
            prompt=prompt,
            negative_prompt=negative,
            seed=seed,
            resolution=resolution
        )
        image_path = self.image_generator.save(image=image, info=info, project_name=project_name)

        # Cache path, Real path, Button state
        return image_path, image_path, gr.update(interactive=True)
    
    def _upscale_last(self, image_path: str):
        path = (image_path or "").strip()
        if not path:
            raise gr.Error("No image generated")

        upscaled_path = self.upscaler.upscale(path)
        return upscaled_path, gr.update(interactive=False)

    def run(self):
        with gr.Blocks() as demo:
            gr.Markdown("Image Generator")

            real_image_path = gr.State(value=None)

            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter your prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=4, value=self.image_generator.default_negative_prompt)

            with gr.Row():
                seed = gr.Textbox(label="Random Seed (default -1)", value="-1", placeholder="e.g. 1234")
                project_name = gr.Textbox(label="Project Name", placeholder="e.g. my_project")

            with gr.Row():
                resolution = gr.Dropdown(
                    choices=list(self.image_generator.RESOLUTION_OPTIONS.keys()),
                    value="1360 × 768 (Landscape)",
                    label="Resolution"
                )

            generate_btn = gr.Button("Generate Image")
            output_image = gr.Image(label="Result", type="filepath", interactive=False)

            gr.Markdown("## Upscale (Real-ESRGAN x4)")
            upscale_btn = gr.Button("Upscale Image", interactive=False)
            upscaled_image = gr.Image(label="Upscaled Result", type="filepath", interactive=False)

            generate_btn.click(
                fn=self._gen_and_save,
                inputs=[prompt, negative_prompt, seed, resolution, project_name],
                outputs=[output_image, real_image_path, upscale_btn]
            )

            upscale_btn.click(
                fn=self._upscale_last,
                inputs=real_image_path,
                outputs=[upscaled_image, upscale_btn]
            )

        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    ArtStation().run()
