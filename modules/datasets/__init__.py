"""Dataset-specific loaders returning the same explicit scene representation."""

from .blender import load_blender_scene
from .llff import load_llff_scene
from .types import SceneData, SceneSplit

__all__ = ['load_scene', 'SceneData', 'SceneSplit']


def load_scene(dataset_path, dataset_type='blender', *, factor=1,
               splits=('train', 'val', 'test'), white_background=None,
               llff_holdout=8, llff_bounds_scale=0.75, llff_recenter=True,
               num_render_poses=80, testskip=1):
    """Load only requested Blender splits, or one LLFF scene with holdout indices.

    LLFF supports pre-downsampled images_<factor> or in-memory Pillow resizing.
    Loading never writes images or invokes ImageMagick/COLMAP.
    """
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError('Dataset downsampling factor must be a positive integer')
    if not splits or len(set(splits)) != len(splits) or set(splits) - {'train', 'val', 'test'}:
        raise ValueError('Requested splits must be unique train, val, or test names')
    options = dict(factor=factor, splits=splits, num_render_poses=num_render_poses)
    if dataset_type == 'blender':
        return load_blender_scene(dataset_path, **options, testskip=testskip,
                                  white_background=True if white_background is None else white_background)
    if dataset_type == 'llff':
        return load_llff_scene(dataset_path, **options, holdout=llff_holdout,
                               bounds_scale=llff_bounds_scale, recenter=llff_recenter,
                               white_background=False if white_background is None else white_background)
    raise ValueError(f'Unknown dataset_type {dataset_type!r}; expected blender or llff')
