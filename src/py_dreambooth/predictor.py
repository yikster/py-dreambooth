import logging
from abc import ABCMeta, abstractmethod
from typing import Final, List, Optional
import boto3
import sagemaker
import torch
from PIL import Image
from sagemaker.huggingface.estimator import HuggingFaceModel
from .model import BaseModel
from .utils.aws_helpers import create_role_if_not_exists
from .utils.image_helpers import decode_base64_image
from .utils.misc import log_or_print

DEFAULT_INSTANCE_TYPE: Final = "ml.g4dn.xlarge"

PYTORCH_VERSION: Final = "2.0.0"
TRANSFORMER_VERSION: Final = "4.28.1"


class BasePredictor(metaclass=ABCMeta):
    def __init__(self, model: BaseModel, logger: Optional[logging.Logger]) -> None:
        self.model = model
        self.logger = logger

    @abstractmethod
    def predict(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: Optional[str],
        num_images_per_prompt: int,
        seed: Optional[int],
    ) -> List[Image.Image]:
        """
        TODO
        """

    def validate_prompt(self, prompt: str) -> bool:
        return (
            hasattr(self.model, "subject_name")
            and hasattr(self.model, "class_name")
            and not (
                self.model.subject_name in prompt.lower()
                and self.model.class_name in prompt.lower()
            )
        )


class LocalPredictor(BasePredictor):
    def __init__(
        self, model: BaseModel, output_dir: str, logger: Optional[logging.Logger] = None
    ):
        super().__init__(model, logger)

        model_components = self.model.load_model(output_dir)
        self.pipeline = model_components["pipeline"]
        self.refiner = model_components.get("refiner")

        log_or_print(
            f"The model has loaded from the directory, '{output_dir}'.", self.logger
        )

    def predict(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 4,
        seed: Optional[int] = None,
        high_noise_frac: float = 0.7,
        cross_attention_scale: float = 0.5,
    ) -> List[Image.Image]:
        if self.validate_prompt(prompt):
            log_or_print(
                "Warning: the subject and class names are not included in the prompt.",
                self.logger,
            )

        generator = (
            None
            if seed is None
            else torch.Generator(device=self.model.device).manual_seed(seed)
        )

        if self.refiner:
            image = self.pipeline(
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
            generated_images = self.refiner(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                denoising_start=high_noise_frac,
            )["images"]

        else:
            generated_images = self.pipeline(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )["images"]

        return generated_images


class AWSPredictor(BasePredictor):
    def __init__(
        self,
        model: BaseModel,
        s3_model_uri: str,
        boto_session: boto3.Session,
        iam_role_name: Optional[str] = None,
        sm_infer_instance_type: Optional[str] = None,
        sm_endpoint_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(model, logger)

        role_name = (
            create_role_if_not_exists(
                boto_session,
                boto_session.region_name,
                logger=self.logger,
            )
            if iam_role_name is None
            else iam_role_name
        )
        infer_instance_type = (
            DEFAULT_INSTANCE_TYPE
            if sm_infer_instance_type is None
            else sm_infer_instance_type
        )
        self.endpoint_name = (
            "py-dreambooth" if sm_endpoint_name is None else sm_endpoint_name
        )

        env = {}
        if model.scheduler_type:
            env.update({"SCHEDULER_TYPE": model.scheduler_type})
        if hasattr(model, "use_ft_vae") and model.use_ft_vae:
            env.update({"USE_FT_VAE": "True"})
        if (
            hasattr(model, "enable_xformers_memory_efficient_attention")
            and model.enable_xformers_memory_efficient_attention
        ):
            env.update({"ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION": "True"})
        if hasattr(model, "use_refiner") and model.use_refiner:
            env.update({"USE_REFINER": "True"})

        sm_session = sagemaker.session.Session(boto_session=boto_session)

        hf_model = HuggingFaceModel(
            role=role_name,
            model_data=s3_model_uri,
            entry_point="inference.py",
            transformers_version=TRANSFORMER_VERSION,
            pytorch_version=PYTORCH_VERSION,
            py_version="py310",
            source_dir=model.infer_source_dir,
            env=None if len(env) == 0 else env,
            sagemaker_session=sm_session,
        )

        self.predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=infer_instance_type,
            endpoint_name=self.endpoint_name,
            volume_size=50,
        )

        log_or_print(
            f"The model has deployed to the endpoint, '{self.endpoint_name}'.",
            self.logger,
        )

    def delete_endpoint(self) -> None:
        self.predictor.delete_endpoint()
        log_or_print(
            f"The endpoint, '{self.endpoint_name}', has been deleted.", self.logger
        )

    def predict(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 4,
        seed: Optional[int] = None,
        high_noise_frac: float = 0.7,
        cross_attention_scale: float = 1.0,
    ) -> List[Image.Image]:
        if self.validate_prompt(prompt):
            log_or_print(
                "Warning: the subject and class names are not included in the prompt.",
                self.logger,
            )

        data = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "high_noise_frac": high_noise_frac,
            "cross_attention_scale": cross_attention_scale,
        }

        if negative_prompt:
            data.update(**{"negative_prompt": negative_prompt})

        if seed:
            data.update(**{"seed": seed})

        generated_images = self.predictor.predict(data)
        generated_images = [
            decode_base64_image(image) for image in generated_images["images"]
        ]

        return generated_images
