import ast
import base64
import os
from io import BytesIO
from typing import Any, Dict, Final, List
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, EulerDiscreteScheduler


class HfModel:
    SDXL_V1_0: Final = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_REFINER_V1_0: Final = "stabilityai/stable-diffusion-xl-refiner-1.0"


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


def model_fn(model_dir: str) -> Dict[str, Any]:
    scheduler_type = os.getenv("SCHEDULER_TYPE", "DDIM")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if scheduler_type.upper() == "DDIM":
        scheduler = DDIMScheduler(**SchedulerConfig.DDIM)
    elif scheduler_type.upper() == "EULERDISCRETE":
        scheduler = EulerDiscreteScheduler(
            **SchedulerConfig.EULER_DISCRETE,
        )
    else:
        scheduler = None
        ValueError("The 'scheduler_type' must be one of 'DDIM' or 'EulerDiscrete'.")

    pipeline = DiffusionPipeline.from_pretrained(
        HfModel.SDXL_V1_0,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    pipeline.load_lora_weights(model_dir)

    if ast.literal_eval(os.environ.get("USE_REFINER", "False")):
        refiner = DiffusionPipeline.from_pretrained(
            HfModel.SDXL_REFINER_V1_0,
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
    else:
        refiner = None

    return {"pipeline": pipeline, "refiner": refiner}


def predict_fn(
    data: Dict[str, Any], model_components: Dict[str, Any]
) -> Dict[str, List[str]]:
    prompt = data.pop("prompt", "")
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", 42)
    high_noise_frac = data.pop("high_noise_frac", 0.7)
    cross_attention_scale = data.pop("cross_attention_scale", 0.5)

    pipeline, refiner = model_components["model"], model_components["refiner"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    negative_prompt = (
        None
        if negative_prompt is None or len(negative_prompt) == 0
        else negative_prompt
    )
    generator = (
        None if seed is None else torch.Generator(device=device).manual_seed(seed)
    )

    if refiner:
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            denoising_end=high_noise_frac,
            generator=generator,
            output_type="latent",
            cross_attention_kwargs={"scale": cross_attention_scale},
        )["images"]
        generated_images = refiner(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
        )["images"]

    else:
        generated_images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            cross_attention_kwargs={"scale": cross_attention_scale},
        )["images"]

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"images": encoded_images}
