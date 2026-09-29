"""Image decoding and downsampling without external programs or dataset writes."""

import numpy as np
from PIL import Image
from pathlib import Path


def validate_rgb_images(images):
    """Check the in-memory contract without implicitly converting encoded pixels."""
    if (
        not isinstance(images, np.ndarray)
        or images.ndim != 4
        or images.shape[-1] != 3
        or any(size < 1 for size in images.shape[:3])
    ):
        raise ValueError("Images must be a nonempty N x H x W x 3 array")
    if not np.issubdtype(images.dtype, np.floating) or any(
        not np.isfinite(image).all() or image.min() < 0 or image.max() > 1
        for image in images
    ):
        raise ValueError(
            "Images must contain finite floating-point RGB values in [0, 1]"
        )


def read_image(path, *, factor=1, white_background=False, reference_blender=False):
    """Decode 8-bit RGB/RGBA PNG or RGB JPEG to float32 RGB in [0, 1]."""
    if type(factor) is not int or factor < 1:
        raise ValueError("Image downsampling factor must be a positive integer")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset image is missing: {path}. Check the metadata path and downloaded files"
        )
    with Image.open(path) as source:
        if source.format not in ("PNG", "JPEG"):
            raise ValueError(f"Expected an 8-bit RGB/RGBA PNG or RGB JPEG: {path}")
        # Pillow silently converts 16-bit RGB PNG to 8-bit. Reject it before decoding.
        if source.format == "PNG":
            with path.open("rb") as file:
                header = file.read(26)
            if len(header) < 26 or header[24] != 8:
                raise ValueError(
                    f"Expected 8-bit PNG channels: {path}. Convert explicitly before loading"
                )
        if source.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Expected RGB or RGBA image, got {source.mode}: {path}")
        width, height = source.size
        if reference_blender:
            # Upstream resizes floating RGBA using INTER_AREA before compositing.
            pixels = np.asarray(source, dtype=np.float32) / 255
            size = (max(1, width // factor), max(1, height // factor))
            if size != source.size:

                def area_weights(original, resized):
                    edges = np.linspace(0, original, resized + 1)
                    starts = np.arange(original)
                    overlap = np.maximum(
                        0,
                        np.minimum(edges[1:, None], starts + 1)
                        - np.maximum(edges[:-1, None], starts),
                    )
                    return (overlap / (original / resized)).astype(np.float32)

                pixels = np.einsum(
                    "yh,hwc,xw->yxc",
                    area_weights(height, size[1]),
                    pixels,
                    area_weights(width, size[0]),
                    optimize=True,
                )
        else:
            size = (max(1, round(width / factor)), max(1, round(height / factor)))
            image = (
                source.resize(size, Image.Resampling.LANCZOS)
                if size != source.size
                else source
            )
            pixels = np.asarray(image, dtype=np.float32) / 255
    if pixels.shape[-1] == 4:
        pixels = (
            pixels[..., :3]
            if reference_blender and not white_background
            else pixels[..., :3] * pixels[..., 3:4]
            + float(white_background) * (1 - pixels[..., 3:4])
        )
    return pixels, (height, width)


def stack_images(images):
    if not images or any(image.shape != images[0].shape for image in images):
        raise ValueError("Dataset images must be nonempty and have matching dimensions")
    return np.stack(images).astype(np.float32, copy=False)
