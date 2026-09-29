"""Scene coordinates used by the networks, independent of ray sampling bounds."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SceneNormalization:
    """Map a world-space cube centered at center with half-extent scale to [-1, 1].

    A uniform scale preserves relative distances. Points outside the cube are
    allowed and are not clipped. Rays and integration intervals stay in world units.
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0

    def __post_init__(self):
        center = tuple(float(value) for value in self.center)
        scale = float(self.scale)
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ValueError("Scene center must contain three finite coordinates")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Scene scale must be finite and positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    def to_dict(self):
        """Return plain values compatible with weights-only checkpoint loading."""
        return {"center": list(self.center), "scale": self.scale}


def resolve_scene_normalization(config, checkpoint=None):
    """Use explicit settings for new runs and saved coordinates for restored runs."""
    transform = (
        SceneNormalization(**checkpoint["scene_normalization"])
        if checkpoint is not None
        else SceneNormalization(
            center=config.get("scene_center", [0.0, 0.0, 0.0]),
            scale=config.get("scene_scale", 1.0),
        )
    )
    if (
        config.get("dataset_type") == "llff" or config.get("model_type") == "nerf"
    ) and transform != SceneNormalization():
        raise ValueError("NeRF and LLFF require identity scene normalization")
    return transform
