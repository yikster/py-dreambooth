import logging
import os
import shutil
import tarfile
import zipfile
from typing import Optional


def compress_dir_to_model_tar_gz(
    src_dir: str,
    tgt_dir: Optional[str] = None,
    tgt_filename: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if tgt_dir is None:
        tgt_dir = src_dir

    if tgt_filename is None:
        tgt_filename = "model.tar.gz"

    src_dir = os.path.abspath(src_dir)
    tgt_dir = os.path.abspath(tgt_dir)
    tgt_file_path = os.path.join(tgt_dir, tgt_filename)

    msg = "The following directories and files will be compressed:"
    log_or_print(msg, logger)

    os.chdir(src_dir)
    with tarfile.open(tgt_file_path, "w:gz") as file:
        for item in os.listdir("."):
            log_or_print(f"Adding {item}", logger)
            file.add(item, arcname=item)

    os.chdir(tgt_dir)


def decompress_file(
    src_file_path: str, tgt_dir: Optional[str] = None, compression: Optional[str] = None
) -> None:
    if tgt_dir is None:
        tgt_dir = os.path.dirname(src_file_path)

    if compression is None:
        compression = "zip"

    if compression == "zip":
        with zipfile.ZipFile(src_file_path) as file:
            file.extractall(tgt_dir)
    elif compression == "tar":
        with tarfile.open(src_file_path) as file:
            file.extractall(tgt_dir)
    else:
        raise ValueError("The argument, 'compression' should be 'zip or 'tar'.")


def delete_dir_with_name(
    root_dir: str, dir_name: str, logger: Optional[logging.Logger] = None
) -> None:
    for root, dirs, _ in os.walk(root_dir, topdown=False):
        for name in dirs:
            if name == dir_name:
                dir_path = os.path.join(root, name)
                shutil.rmtree(dir_path)
                msg = f"The deleted directory is '{dir_path}'."
                log_or_print(msg, logger)


def log_or_print(msg: str, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(msg)
    else:
        print(msg)
