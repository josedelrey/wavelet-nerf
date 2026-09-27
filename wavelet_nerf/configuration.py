"""Strict YAML loading and typed configuration validation."""

import math
from pathlib import Path
import re

import yaml


_COMMON = {
    "dataset_path": "./datasets/lego",
    "seed": 42,
    "deterministic": False,
    "dataset_type": "blender",
    "dataset_factor": 1,
    "llff_holdout": 8,
    "llff_bounds_scale": 0.75,
    "llff_recenter": True,
    "white_background": True,
    "near": 2.0,
    "far": 6.0,
    "num_random_rays": 1024,
    "num_samples": 256,
    "num_samples_eval": 256,
    "chunk_size": 8192,
    "learning_rate": 5e-4,
    "lr_decay": 150.0,
    "lr_decay_factor": 0.1,
    "lr_min": 1e-5,
    "num_iters": 150000,
    "save_interval": 5000,
    "log_interval": 10,
    "val_interval": 1000,
    "first_step_render": False,
    "log_root": "./logs",
    "save_path": "./models",
    "num_render_poses": 40,
    "device": "auto",
    "compile_model": False,
    "num_workers": 0,
    "netchunk": 65536,
}
_MODEL = {
    "nerf": {
        "pos_encoding_dim": 10,
        "dir_encoding_dim": 4,
        "hidden_dim": 256,
        "num_importance": 128,
        "perturb": 1.0,
        "lindisp": False,
        "raw_noise_std": 0.0,
        "white_background": True,
        "no_batching": True,
        "precrop_iters": 0,
        "precrop_frac": 0.5,
        "half_res": False,
        "testskip": 8,
    },
    "siren": {
        "num_layers": 8,
        "siren_hidden_dim": 256,
        "siren_dir_encoding_dim": 4,
        "sigma_mul": 10.0,
        "rgb_mul": 1.0,
        "w0": 30.0,
        "hidden_w0": 1.0,
    },
    "wavelet": {
        "wave_in_features": 3,
        "wave_hidden_dim": 256,
        "wave_num_layers": 8,
        "wave_dir_encoding_dim": 4,
        "input_scale": 256.0,
        "weight_scale": 1.0,
        "alpha": 6.0,
        "beta": 0.5,
        "omega0": 5.0,
        "normalized": True,
    },
}

_EXTRA = {
    "model_type": "nerf",
    "experiment_name": "",
    "scene_center": [0.0, 0.0, 0.0],
    "scene_scale": 1.0,
    "render_orbit_elevation": -30.0,
    "render_orbit_radius": 4.0,
    "render_spiral_rotations": 2.0,
    "render_spiral_zrate": 0.5,
    "render_spiral_radius_scale": 1.0,
}
_TYPES = {
    key: type(value)
    for group in (_COMMON, *_MODEL.values(), _EXTRA)
    for key, value in group.items()
}
# These loader options apply to every model.
_COMMON.update(half_res=False, testskip=1)
_COMMON.update(
    {key: value for key, value in _EXTRA.items() if key.startswith("render_")}
)
_POSITIVE_INTS = {
    "dataset_factor",
    "llff_holdout",
    "num_random_rays",
    "num_samples",
    "num_samples_eval",
    "chunk_size",
    "num_iters",
    "save_interval",
    "log_interval",
    "val_interval",
    "num_render_poses",
    "netchunk",
    "testskip",
    "hidden_dim",
    "siren_hidden_dim",
    "wave_hidden_dim",
    "num_layers",
    "wave_num_layers",
    "wave_in_features",
}
_NONNEGATIVE_INTS = {
    "seed",
    "num_importance",
    "precrop_iters",
    "pos_encoding_dim",
    "dir_encoding_dim",
    "siren_dir_encoding_dim",
    "wave_dir_encoding_dim",
    "num_workers",
}
_POSITIVE_FLOATS = {
    "learning_rate",
    "llff_bounds_scale",
    "scene_scale",
    "sigma_mul",
    "rgb_mul",
    "w0",
    "hidden_w0",
    "input_scale",
    "weight_scale",
    "alpha",
    "beta",
    "omega0",
    "render_orbit_radius",
    "render_spiral_rotations",
    "render_spiral_radius_scale",
}
_NONNEGATIVE_FLOATS = {"lr_decay", "lr_min", "raw_noise_std", "perturb"}


class _UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError(
                    f"Configuration keys must be strings at line {key_node.start_mark.line + 1}"
                )
            if key in result:
                raise ValueError(
                    f"Duplicate configuration key {key!r} at line {key_node.start_mark.line + 1}"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Avoid YAML 1.1's implicit yes/no booleans and octal/sexagesimal numbers.
_UniqueSafeLoader.yaml_implicit_resolvers = {
    key: [
        (tag, pattern)
        for tag, pattern in resolvers
        if tag
        not in {
            "tag:yaml.org,2002:bool",
            "tag:yaml.org,2002:int",
            "tag:yaml.org,2002:float",
        }
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_UniqueSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false|True|False|TRUE|FALSE)$"),
    list("tTfF"),
)
_UniqueSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
    list("-+0123456789"),
)
_UniqueSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+)"
        r"(?:[eE][-+]?[0-9]+)?|[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def normalize_config(config):
    """Validate native, typed configuration fields."""
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("Configuration must be a mapping with string keys")
    unknown = set(config) - _TYPES.keys()
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    result = {}
    for key, original in config.items():
        value, expected = original, _TYPES[key]
        if key in ("near", "far") and value == "auto":
            result[key] = value
            continue
        if expected is float:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
            value = float(value)
        elif expected is list:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(item)
                    for item in value
                )
            ):
                raise ValueError(f"{key} must be a list of three finite numbers")
            value = list(map(float, value))
        elif type(value) is not expected:
            raise ValueError(
                f"{key} must be {expected.__name__}, got {type(value).__name__}"
            )
        if key in _POSITIVE_INTS | _POSITIVE_FLOATS and value <= 0:
            raise ValueError(f"{key} must be positive")
        if key in _NONNEGATIVE_INTS | _NONNEGATIVE_FLOATS and value < 0:
            raise ValueError(f"{key} must be nonnegative")
        if key in ("hidden_dim", "siren_hidden_dim", "wave_hidden_dim") and value < 2:
            raise ValueError(f"{key} must be at least 2")
        if key == "wave_in_features" and value != 3:
            raise ValueError("wave_in_features must be 3 for 3D geometry")
        if key == "llff_holdout" and value < 2:
            raise ValueError("llff_holdout must be at least 2")
        if key == "seed" and value > 2**32 - 1:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if key in ("precrop_frac", "lr_decay_factor") and not 0 < value <= 1:
            raise ValueError(f"{key} must lie in (0, 1]")
        if (
            key in ("dataset_path", "save_path", "log_root", "experiment_name")
            and not value.strip()
        ):
            raise ValueError(f"{key} must be a nonempty string")
        if key == "experiment_name" and (
            value in (".", "..") or "/" in value or "\\" in value
        ):
            raise ValueError("experiment_name must be a single directory name")
        if key in ("model_type", "dataset_type"):
            value = value.lower()
        if key == "model_type":
            if value not in _MODEL:
                raise ValueError(f"Invalid model_type: {value!r}")
        if key == "dataset_type" and value not in ("blender", "llff"):
            raise ValueError("dataset_type must be blender or llff")
        if key == "device" and not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", value):
            raise ValueError(
                "device must be auto, cpu, cuda or cuda:<index>; MPS is not supported"
            )
        result[key] = value
    return result


def parse_config(config_path):
    """Read one strict YAML mapping; files may contain partial evaluation overrides."""
    path = Path(config_path)
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Configuration files must use YAML (.yaml or .yml)")
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.load(file, Loader=_UniqueSafeLoader)
        return normalize_config(config)
    except (yaml.YAMLError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def validate_resolved_config(config, *, training=False):
    config = normalize_config(config)
    model_keys = set().union(*_MODEL.values())
    unsupported = config.keys() & (
        model_keys - _MODEL[config["model_type"]].keys() - _COMMON.keys()
    )
    if unsupported:
        raise ValueError(
            f"Configuration keys are not supported by {config['model_type']}: "
            f"{', '.join(sorted(unsupported))}"
        )
    if training and not config.get("experiment_name"):
        raise ValueError("Training configuration requires experiment_name")
    near, far = config["near"], config["far"]
    if not near < far or (config["dataset_type"] == "blender" and near <= 0):
        raise ValueError("Bounds require near < far, with near > 0 for Blender")
    if config["dataset_type"] == "llff":
        if (near, far) != (0.0, 1.0):
            raise ValueError("LLFF NDC requires near = 0 and far = 1")
        if config.get("lindisp", False) or config["white_background"]:
            raise ValueError(
                "LLFF NDC requires lindisp = false and white_background = false"
            )
        if (
            config.get("scene_center", [0.0, 0.0, 0.0]) != [0.0, 0.0, 0.0]
            or config.get("scene_scale", 1.0) != 1.0
        ):
            raise ValueError("LLFF NDC requires identity scene_center and scene_scale")
    if config["lr_min"] > config["learning_rate"]:
        raise ValueError("lr_min cannot exceed learning_rate")
    if config["model_type"] == "nerf":
        if (
            config.get("scene_center", [0.0, 0.0, 0.0]) != [0.0, 0.0, 0.0]
            or config.get("scene_scale", 1.0) != 1.0
        ):
            raise ValueError(
                "Reference NeRF requires identity scene_center and scene_scale"
            )
        minimum = 3 if config["num_importance"] else 2
        if min(config["num_samples"], config["num_samples_eval"]) < minimum:
            raise ValueError(
                f"Reference sampling needs at least {minimum} coarse samples"
            )
        if training and config["lr_min"] != 0:
            raise ValueError(
                "Reference NeRF uses exponential decay without an lr_min floor; set lr_min = 0"
            )
    return config
