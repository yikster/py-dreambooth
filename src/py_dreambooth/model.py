import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Final, List, Optional
import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)
from diffusers.models import AutoencoderKL


class HfModel:
    """
    A class to store HuggingFace Hub model IDs
    """

    SD_V1_4: Final = "CompVis/stable-diffusion-v1-4"
    SD_V1_5: Final = "runwayml/stable-diffusion-v1-5"
    SD_V2_1: Final = "stabilityai/stable-diffusion-2-1-base"
    SD_VAE: Final = "stabilityai/sd-vae-ft-mse"
    SDXL_V1_0: Final = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_REFINER_V1_0: Final = "stabilityai/stable-diffusion-xl-refiner-1.0"
    SDXL_VAE: Final = "madebyollin/sdxl-vae-fp16-fix"


class CodeFilename:
    """
    A class to store script file names that will be used for training
    """

    SD_DREAMBOOTH: Final = "train_sd_dreambooth.py"
    SDXL_DREAMBOOTH_LORA: Final = "train_sdxl_dreambooth_lora.py"


class SourceDir:
    """
    A class to store source directory names that will be used to SageMaker Endpoint
    """

    SD_DREAMBOOTH: Final = "sd_dreambooth"
    SDXL_DREAMBOOTH_LORA: Final = "sdxl_dreambooth_lora"


class SchedulerConfig:
    """
    A class to store scheduler configuration values for inference
    """

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


class BaseModel(metaclass=ABCMeta):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        seed: Optional[int],
        resolution: int,
        center_crop: bool,
        train_batch_size: int,
        num_train_epochs: int,
        max_train_steps: Optional[int],
        learning_rate: float,
        scheduler_type: Optional[str],
        compress_output: bool,
    ):
        if pretrained_model_name_or_path in (
            HfModel.SD_V1_4,
            HfModel.SD_V1_5,
        ):
            resolution = min(resolution, 512)
        elif pretrained_model_name_or_path == HfModel.SD_V2_1:
            resolution = min(resolution, 768)
        else:
            resolution = min(resolution, 1024)

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.data_dir = None
        self.output_dir = None
        self.seed = seed
        self.resolution = resolution
        self.center_crop = center_crop
        self.train_batch_size = train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.report_to = None
        self.scheduler_type = "DDIM" if scheduler_type is None else scheduler_type
        self.compress_output = compress_output

        self.train_code_path = None
        self.infer_source_dir = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_abs_path(*args) -> str:
        return os.path.join(os.path.dirname(__file__), *args)

    @abstractmethod
    def get_arguments(self) -> List[str]:
        """
        TODO
        """

    @abstractmethod
    def load_model(self, output_dir: str) -> Dict[str, Any]:
        """
        TODO
        """

    def make_command(self, config_path: Optional[str] = None) -> str:
        launch_arguments = (
            "" if config_path is None else f"--config_file {config_path} "
        )

        command = f"accelerate launch {launch_arguments}{self.train_code_path} "
        command += " ".join(self.get_arguments())
        return command

    def set_members(self, **kwarg) -> "BaseModel":
        for key, value in kwarg.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


class SdDreamboothModel(BaseModel):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        subject_name: Optional[str] = None,
        class_name: Optional[str] = None,
        with_prior_preservation: bool = True,
        seed: Optional[int] = None,
        resolution: int = 768,
        center_crop: bool = False,
        train_text_encoder: bool = True,
        train_batch_size: int = 1,
        num_train_epochs: int = 1,
        max_train_steps: Optional[int] = 1000,
        learning_rate: float = 2e-06,
        validation_prompt: Optional[str] = None,
        reduce_gpu_memory_usage: bool = True,
        scheduler_type: Optional[str] = None,
        use_ft_vae: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        compress_output: bool = False,
    ):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = HfModel.SD_V2_1
        if subject_name is None:
            subject_name = "sks"
        if class_name is None:
            class_name = "person"

        super().__init__(
            pretrained_model_name_or_path,
            seed,
            resolution,
            center_crop,
            train_batch_size,
            num_train_epochs,
            max_train_steps,
            learning_rate,
            scheduler_type,
            compress_output,
        )

        self.subject_name = subject_name
        self.class_name = class_name
        self.with_prior_preservation = with_prior_preservation
        self.train_text_encoder = train_text_encoder
        self.validation_prompt = validation_prompt
        self.reduce_gpu_memory_usage = reduce_gpu_memory_usage
        self.use_ft_vae = use_ft_vae
        self.enable_xformers_memory_efficient_attention = (
            enable_xformers_memory_efficient_attention
        )

        self.train_code_path = self.get_abs_path(
            "scripts",
            "train",
            CodeFilename.SD_DREAMBOOTH,
        )
        self.infer_source_dir = self.get_abs_path(
            "scripts", "infer", SourceDir.SD_DREAMBOOTH
        )

    def load_model(
        self,
        output_dir: str,
    ) -> Dict[str, Any]:
        if self.scheduler_type.upper() == "DDIM":
            scheduler = DDIMScheduler(**SchedulerConfig.DDIM)
        elif self.scheduler_type.upper() == "EULERDISCRETE":
            scheduler = EulerDiscreteScheduler(
                **SchedulerConfig.EULER_DISCRETE,
            )
        else:
            scheduler = None
            ValueError("The 'scheduler_type' must be one of 'DDIM' or 'EulerDiscrete'.")

        pipeline = StableDiffusionPipeline.from_pretrained(
            output_dir,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)

        if self.use_ft_vae:
            pipeline.vae = AutoencoderKL.from_pretrained(
                HfModel.SD_VAE, torch_dtype=torch.float16
            ).to(self.device)

        if self.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        return {"pipeline": pipeline}

    def get_arguments(self) -> List[str]:
        assert (
            self.data_dir or self.output_dir
        ), "'data_dir' or 'output_dir' is required."

        instance_prompt = f"a photo of {self.subject_name} {self.class_name}"
        class_prompt = f"a photo of {self.class_name}"

        if self.validation_prompt is None:
            self.validation_prompt = (
                f"{instance_prompt} with Eiffel Tower in the background"
            )

        arguments = [
            "--pretrained_model_name_or_path",
            self.pretrained_model_name_or_path,
            "--instance_data_dir",
            self.data_dir,
            "--instance_prompt",
            f"'{instance_prompt}'",
            "--num_class_images",
            200,
            "--output_dir",
            self.output_dir,
            "--resolution",
            self.resolution,
            "--train_batch_size",
            self.train_batch_size,
            "--learning_rate",
            self.learning_rate,
            "--lr_scheduler",
            "constant",
            "--lr_warmup_steps",
            0,
            "--validation_prompt",
            f"'{self.validation_prompt}'",
            "--compress_output",
            self.compress_output,
        ]

        if self.with_prior_preservation:
            arguments += [
                "--with_prior_preservation",
                "True",
                "--prior_loss_weight",
                "1.0",
                "--class_prompt",
                f"'{class_prompt}'",
            ]

            if self.reduce_gpu_memory_usage:
                arguments += ["--gradient_accumulation_steps", "1"]

            if self.train_text_encoder:
                arguments += ["--train_text_encoder", "True"]

                if self.reduce_gpu_memory_usage:
                    arguments += [
                        "--gradient_checkpointing",
                        "True",
                        "--use_8bit_adam",
                        "True",
                        "--enable_xformers_memory_efficient_attention",
                        "True",
                        "--mixed_precision",
                        "fp16",
                        "--set_grads_to_none",
                        "True",
                    ]

        if self.seed:
            arguments += [
                "--seed",
                self.seed,
            ]

        if self.center_crop:
            arguments += ["--center_crop"]

        if self.max_train_steps:
            arguments += [
                "--max_train_steps",
                self.max_train_steps,
            ]

        if self.report_to:
            arguments += [
                "--report_to",
                self.report_to,
            ]

        return list(map(str, arguments))


class SdxlDreamboothLoraModel(BaseModel):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        subject_name: Optional[str] = None,
        class_name: Optional[str] = None,
        with_prior_preservation: bool = True,
        seed: Optional[int] = None,
        resolution: int = 1024,
        center_crop: bool = False,
        train_text_encoder: bool = True,
        train_batch_size: int = 1,
        num_train_epochs: int = 1,
        max_train_steps: Optional[int] = 1000,
        learning_rate: float = 1e-5,
        validation_prompt: Optional[str] = None,
        reduce_gpu_memory_usage: bool = True,
        scheduler_type: Optional[str] = None,
        use_refiner: bool = False,
        compress_output: bool = False,
    ):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = HfModel.SDXL_V1_0
        if subject_name is None:
            subject_name = "sks"
        if class_name is None:
            class_name = "person"

        super().__init__(
            pretrained_model_name_or_path,
            seed,
            resolution,
            center_crop,
            train_batch_size,
            num_train_epochs,
            max_train_steps,
            learning_rate,
            scheduler_type,
            compress_output,
        )

        # self.pretrained_vae_model_name_or_path = HfModel.SDXL_VAE
        self.pretrained_vae_model_name_or_path = None
        self.subject_name = subject_name
        self.class_name = class_name
        self.with_prior_preservation = with_prior_preservation
        self.train_text_encoder = train_text_encoder
        self.validation_prompt = validation_prompt
        self.reduce_gpu_memory_usage = reduce_gpu_memory_usage
        self.use_refiner = use_refiner

        self.train_code_path = self.get_abs_path(
            "scripts",
            "train",
            CodeFilename.SDXL_DREAMBOOTH_LORA,
        )
        self.infer_source_dir = self.get_abs_path(
            "scripts",
            "infer",
            SourceDir.SDXL_DREAMBOOTH_LORA,
        )

    def load_model(
        self,
        output_dir: str,
    ) -> Dict[str, Any]:
        if self.scheduler_type.upper() == "DDIM":
            scheduler = DDIMScheduler(**SchedulerConfig.DDIM)
        elif self.scheduler_type.upper() == "EULERDISCRETE":
            scheduler = EulerDiscreteScheduler(
                **SchedulerConfig.EULER_DISCRETE,
            )
        else:
            scheduler = None
            ValueError("The 'scheduler_type' must be one of 'DDIM' or 'EulerDiscrete'.")

        pipeline = DiffusionPipeline.from_pretrained(
            HfModel.SDXL_V1_0,
            scheduler=scheduler,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)

        pipeline.load_lora_weights(output_dir)

        if self.use_refiner:
            refiner = DiffusionPipeline.from_pretrained(
                HfModel.SDXL_REFINER_V1_0,
                vae=pipeline.vae,
                text_encoder_2=pipeline.text_encoder_2,
                variant="fp16",
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            refiner = None

        return {"pipeline": pipeline, "refiner": refiner}

    def get_arguments(self) -> List[str]:
        assert (
            self.data_dir or self.output_dir
        ), "'data_dir' or 'output_dir' is required."

        instance_prompt = f"a photo of {self.subject_name} {self.class_name}"
        class_prompt = f"a photo of {self.class_name}"

        if self.validation_prompt is None:
            self.validation_prompt = (
                f"{instance_prompt} with Eiffel Tower in the background"
            )

        arguments = [
            "--pretrained_model_name_or_path",
            self.pretrained_model_name_or_path,
            # "--pretrained_vae_model_name_or_path",
            # self.pretrained_vae_model_name_or_path,
            "--instance_data_dir",
            self.data_dir,
            "--instance_prompt",
            f"'{instance_prompt}'",
            "--num_class_images",
            200,
            "--output_dir",
            self.output_dir,
            "--resolution",
            self.resolution,
            "--train_batch_size",
            self.train_batch_size,
            "--sample_batch_size",
            2,
            "--learning_rate",
            self.learning_rate,
            "--lr_scheduler",
            "constant",
            "--lr_warmup_steps",
            0,
            "--validation_prompt",
            f"'{self.validation_prompt}'",
            "--compress_output",
            self.compress_output,
        ]

        if self.with_prior_preservation:
            arguments += [
                "--with_prior_preservation",
                "True",
                "--prior_loss_weight",
                "1.0",
                "--class_prompt",
                f"'{class_prompt}'",
            ]

            if self.reduce_gpu_memory_usage:
                arguments += ["--gradient_accumulation_steps", "4"]

            if self.train_text_encoder:
                arguments += ["--train_text_encoder", "True"]

                if self.reduce_gpu_memory_usage:
                    arguments += [
                        "--gradient_checkpointing",
                        "True",
                        "--use_8bit_adam",
                        "True",
                        "--enable_xformers_memory_efficient_attention",
                        "True",
                        "--mixed_precision",
                        "fp16",
                    ]

        if self.seed:
            arguments += [
                "--seed",
                self.seed,
            ]

        if self.center_crop:
            arguments += ["--center_crop"]

        if self.max_train_steps:
            arguments += [
                "--max_train_steps",
                self.max_train_steps,
            ]

        if self.report_to:
            arguments += [
                "--report_to",
                self.report_to,
            ]

        return list(map(str, arguments))
