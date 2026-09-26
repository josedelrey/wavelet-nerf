"""Image decoding and downsampling without external programs or dataset writes."""

import numpy as np
from PIL import Image


def read_image(path, *, factor=1, white_background=False, reference_blender=False):
    with Image.open(path) as source:
        if source.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'Expected RGB or RGBA image, got {source.mode}: {path}')
        width, height = source.size
        if reference_blender:
            # Upstream resizes floating RGBA using INTER_AREA before compositing.
            pixels = np.asarray(source, dtype=np.float32) / 255
            size = (max(1, width // factor), max(1, height // factor))
            if size != source.size:
                def area_weights(original, resized):
                    edges = np.linspace(0, original, resized + 1)
                    starts = np.arange(original)
                    overlap = np.maximum(0, np.minimum(edges[1:, None], starts + 1)
                                         - np.maximum(edges[:-1, None], starts))
                    return (overlap / (original / resized)).astype(np.float32)
                pixels = np.einsum('yh,hwc,xw->yxc', area_weights(height, size[1]), pixels,
                                   area_weights(width, size[0]), optimize=True)
        else:
            size = (max(1, round(width / factor)), max(1, round(height / factor)))
            image = source.resize(size, Image.Resampling.LANCZOS) if size != source.size else source
            pixels = np.asarray(image, dtype=np.float32) / 255
    if pixels.shape[-1] == 4:
        pixels = (pixels[..., :3] if reference_blender and not white_background else
                  pixels[..., :3] * pixels[..., 3:4] + float(white_background) * (1 - pixels[..., 3:4]))
    return pixels, (height, width)


def stack_images(images):
    if not images or any(image.shape != images[0].shape for image in images):
        raise ValueError('Dataset images must be nonempty and have matching dimensions')
    return np.stack(images).astype(np.float32, copy=False)
