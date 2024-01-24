import torch
from typing import List
from PIL import Image
from cog import BasePredictor, Input, Path
from .style_template import styles
import random
import numpy as np

STYLE_NAMES = list(styles.keys())

def resize_image(image, max_width, max_height):
    """
    Resize an image to a specific height while maintaining the aspect ratio and ensuring
    that neither width nor height exceed the specified maximum values.

    Args:
        image (PIL.Image.Image): The input image.
        max_width (int): The maximum allowable width for the resized image.
        max_height (int): The maximum allowable height for the resized image.

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get the original image dimensions
    original_width, original_height = image.size

    # Calculate the new dimensions to maintain the aspect ratio and not exceed the maximum values
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # Choose the smallest ratio to ensure that neither width nor height exceeds the maximum
    resize_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height
    new_width = int(original_width * resize_ratio)
    new_height = int(original_height * resize_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        from .app import generate_image
        self.generate_image = generate_image

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt",),
        negative_prompt: str = Input(
            description="Negative prompt",
        ),
        num_inference_steps: int = Input(description="Steps", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=5.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        face_image: Path = Input(
            description="face image", default=None
        ),
        pose_image: Path = Input(
            description="pose image", default=None
        ),
        style: str = Input(
            default="(No style)",
            choices=styles,
            description="",
        ),
        enhance_face_region: bool = Input(
            description="",
            default=True,
        ),
        identitynet_strength_ratio: float = Input(
            description="for fedility",
            default=0.8,
            ge=0.1,
            le=2.0,
        ),
        adapter_strength_ratio: float = Input(
            description="for fedility",
            default=0.8,
            ge=0.1,
            le=2.0,
        ),
        max_width: int = Input(
            description="Max width/Resolution of image",
            default=512,
        ),
        max_height: int = Input(
            description="Max height/Resolution of image",
            default=512,
        ),
    ) -> List[Path]:
        
        if not seed:
            seed = random.randint(100, np.iinfo(np.int32).max)
        
        face_image= resize_image(face_image, max_width, max_height)

        if pose_image:
            pose_image= resize_image(pose_image, max_width, max_height)
        
        images, _ = self.generate_image(
            face_image=[face_image],
            pose_image= [pose_image],
            prompt=prompt,
            negative_prompt=negative_prompt,
            style_name=style,
            enhance_face_region=enhance_face_region,
            num_steps=num_inference_steps,
            identitynet_strength_ratio=identitynet_strength_ratio,
            adapter_strength_ratio=adapter_strength_ratio,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        output_path = f"/tmp/output_{seed}.png"
        images[0].save(output_path)

        output_paths= [Path(output_path)]

        return output_paths
