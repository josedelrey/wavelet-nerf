"""Scene coordinates used by the networks, independent of ray sampling bounds."""

from dataclasses import dataclass
import math
import warnings


@dataclass(frozen=True)
class SceneNormalization:
    """Map a world-space cube centered at center with half-extent scale to [-1, 1].

    A uniform scale preserves relative distances. Points outside the cube are
    allowed and are not clipped; rays and integration intervals stay in world units.
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0

    def __post_init__(self):
        center = tuple(float(value) for value in self.center)
        scale = float(self.scale)
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ValueError('Scene center must contain three finite coordinates')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Scene scale must be finite and positive')
        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'scale', scale)

    def to_dict(self):
        """Return plain values compatible with weights-only checkpoint loading."""
        return {'center': list(self.center), 'scale': self.scale}


def resolve_scene_normalization(config, checkpoint=None):
    """Use saved coordinates when restoring, or explicit scene settings for new runs."""
    if config.get('dataset_type') == 'llff':
        transform = (SceneNormalization(**checkpoint['scene_normalization'])
                     if checkpoint is not None and 'scene_normalization' in checkpoint else
                     SceneNormalization(
                         center=tuple(float(value) for value in config.get('scene_center', '0, 0, 0').split(',')),
                         scale=float(config.get('scene_scale', 1.0)),
                     ))
        if transform != SceneNormalization():
            raise ValueError('Reference LLFF uses NDC coordinates directly: scene_center = 0, 0, 0; scene_scale = 1')
        return transform
    if checkpoint is not None:
        if 'scene_normalization' in checkpoint:
            return SceneNormalization(**checkpoint['scene_normalization'])

        # Old networks learned this affine map. Preserve it rather than silently
        # interpreting their weights in the new scene coordinate system.
        near = float(config.get('near', 2.0))
        far = float(config.get('far', 6.0))
        if not math.isfinite(near) or not math.isfinite(far) or far <= near:
            raise ValueError('Legacy checkpoint normalization requires finite near < far')
        warnings.warn(
            'Checkpoint has no scene normalization; preserving the legacy mapping '
            'using near/far from the supplied config. These must match the original '
            'training bounds. New experiments should be retrained with explicit '
            'scene normalization.',
            UserWarning,
            stacklevel=2,
        )
        return SceneNormalization(center=((near + far) / 2,) * 3, scale=(far - near) / 2)

    center = tuple(float(value) for value in config.get('scene_center', '0, 0, 0').split(','))
    return SceneNormalization(center=center, scale=float(config.get('scene_scale', 1.0)))
