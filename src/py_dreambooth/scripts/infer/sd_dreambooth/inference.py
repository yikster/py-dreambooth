import ast
import base64
import os
from io import BytesIO
from typing import Any, Dict, Final, List, Union
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL


class HfModel:
    SD_VAE: Final = "stabilityai/sd-vae-ft-mse"


class SchedulerConfig:
    DDIM: Final = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": True,
        "steps_offset": 1,
    }
    EULER_DISCRETE: Final = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "use_karras_sigmas": True,
        "steps_offset": 1,
    }


def model_fn(model_dir: str) -> Any:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler_type = os.getenv("SCHEDULER_TYPE", "DDIM")

    if scheduler_type.upper() == "DDIM":
        scheduler = DDIMScheduler(**SchedulerConfig.DDIM)
    elif scheduler_type.upper() == "EULERDISCRETE":
        scheduler = EulerDiscreteScheduler(
            **SchedulerConfig.EULER_DISCRETE,
        )
    else:
        scheduler = None
        ValueError("The 'scheduler_type' must be one of 'DDIM' or 'EulerDiscrete'.")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_dir,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    if ast.literal_eval(os.environ.get("USE_FT_VAE", "False")):
        pipeline.vae = AutoencoderKL.from_pretrained(
            HfModel.SD_VAE, torch_dtype=torch.float16
        ).to(device)

    if ast.literal_eval(
        os.getenv("ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION", "False")
    ):
        pipeline.enable_xformers_memory_efficient_attention()

    return {"pipeline": pipeline}


def predict_fn(
    data: Dict[str, Union[int, float, str]], model_components: Dict[str, Any]
) -> Dict[str, List[str]]:
    prompt = data.pop("prompt", data)
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", None)

    pipeline = model_components["pipeline"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    negative_prompt = (
        None
        if negative_prompt is None or len(negative_prompt) == 0
        else negative_prompt
    )
    generator = (
        None if seed is None else torch.Generator(device=device).manual_seed(seed)
    )

    generated_images = pipeline(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )["images"]

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"images": encoded_images}
