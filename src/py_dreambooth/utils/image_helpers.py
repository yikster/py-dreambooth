import base64
import os
from glob import glob
from itertools import chain
from typing import List, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from autocrop import Cropper


def decode_base64_image(image_string: str) -> Image.Image:
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


def encode_base64_image(file_name: str) -> str:
    with open(file_name, "rb") as image:
        image_string = base64.b64encode(bytearray(image.read())).decode()
    return image_string


def detect_face_and_resize_image(
    image_path: str, tgt_width: int, tgt_height: int
) -> Image:
    cropper = Cropper(width=tgt_width, height=tgt_height)
    cropped_array = cropper.crop(image_path)

    if cropped_array is None:
        msg = f"No faces detected in the image '{image_path.split(os.path.sep)[-1]}'."
        raise RuntimeError(msg)

    return Image.fromarray(cropped_array)


def display_images(
    images: List[Image.Image],
    n_columns: int = 3,
    fig_size: int = 20,
) -> None:
    n_columns = min(len(images), n_columns)
    quotient, remainder = divmod(len(images), n_columns)
    if remainder > 0:
        quotient += 1
    width, height = images[0].size
    plt.figure(figsize=(fig_size, fig_size / n_columns * quotient * height / width))

    for i, image in enumerate(images):
        plt.subplot(quotient, n_columns, i + 1)
        plt.axis("off")
        plt.imshow(image, aspect="auto")

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()


def display_image_grid(images: List[Image.Image], n_columns: int = 3) -> Image.Image:
    n_columns = min(len(images), n_columns)
    quotient, remainder = divmod(len(images), n_columns)
    if remainder > 0:
        quotient += 1
    width, height = images[0].size
    grid = Image.new(
        "RGB",
        size=(
            n_columns * width,
            quotient * height,
        ),
    )

    for i, image in enumerate(images):
        grid.paste(image, box=(i % n_columns * width, i // n_columns * height))

    return grid


def get_image_paths(images_dir: str) -> List[str]:
    return list(
        chain(
            glob(os.path.join(images_dir, "*.[jJ][pP]*[Gg]")),
            glob(os.path.join(images_dir, "*.[Pp][Nn][Gg]")),
        )
    )


def resize_and_center_crop_image(
    image_path: str, tgt_width: int, tgt_height: int
) -> Image:
    image = Image.open(image_path).convert("RGB")
    src_width, src_height = image.size

    if src_width > src_height:
        left = (src_width - src_height) / 2
        top = 0
        right = (src_width + src_height) / 2
        bottom = src_height
    else:
        top = (src_height - src_width) / 2
        left = 0
        bottom = (src_height + src_width) / 2
        right = src_width

    image = image.crop((left, top, right, bottom))
    image = image.resize((tgt_width, tgt_height))

    return image


def validate_dir(tgt_dir: str) -> None:
    if not os.path.exists(tgt_dir) and len(get_image_paths(tgt_dir)) == 0:
        raise ValueError(f"The directory '{tgt_dir}' does not exist or is empty.")
