import gradio as gr

from generators.hidream import HiDream
from generators.base import BaseImageGenerator


def make_image_generator() -> BaseImageGenerator:
    # Todo: variable models
    return HiDream(model_type="full")

class ArtStation:
    def __init__(self):
        self.model = make_image_generator()

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
        image, info = self.model.generate(
            prompt=prompt,
            negative_prompt=negative,
            seed=seed,
            resolution=resolution
        )
        image_path = self.model.save(image=image, info=info, project_name=project_name)

        return image_path

    def run(self):
        with gr.Blocks() as demo:
            gr.Markdown("Image Generator")

            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter your prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=4, value=self.model.default_negative_prompt)

            with gr.Row():
                seed = gr.Textbox(label="Random Seed (default -1)", value="-1", placeholder="e.g. 1234")
                project_name = gr.Textbox(label="Project Name", placeholder="e.g. my_project")

            with gr.Row():
                resolution = gr.Dropdown(
                    choices=list(self.model.RESOLUTION_OPTIONS.keys()),
                    value="1360 × 768 (Landscape)",
                    label="Resolution"
                )

            generate_btn = gr.Button("Generate Image")
            output_image = gr.Image(label="Result")

            generate_btn.click(
                fn=self._gen_and_save,
                inputs=[prompt, negative_prompt, seed, resolution, project_name],
                outputs=output_image
            )

        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    ArtStation().run()
