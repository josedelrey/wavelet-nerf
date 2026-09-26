# Wavelet-NeRF


[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes. This project is a PyTorch implementation of NeRF, extended with [SIREN-based](https://arxiv.org/abs/2006.09661) and [MFN-based](https://arxiv.org/abs/2011.13961) NeRF variants. The code is based on the authors' original TensorFlow implementation [here](https://github.com/bmild/nerf).

![Lego novel-view render generated with this repository](imgs/nerf_lego.gif)

Lego render created by José del Rey using this repository and the NeRF Synthetic
dataset.

## Installation

Use Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run the following commands from the repository root:

```bash
git clone https://github.com/josedelrey/wavelet-nerf.git
cd wavelet-nerf
uv sync --locked
```

`uv sync --locked` installs the versions recorded in `uv.lock` into `.venv/`,
including the development linter. Add `--no-dev` to both `uv sync` and `uv run`
commands to omit development tools. `uv run` uses this environment without
activating it.

Dependencies and project metadata are defined in `pyproject.toml`. The project
is configured with `package = false`: uv manages its dependencies without
building or installing this repository as a distribution.

The default installs CPU-only PyTorch from the official CPU index. For NVIDIA
CUDA 13.0 on Linux or Windows, select the `cu130` group instead:

```bash
uv sync --locked --no-group cpu --group cu130
uv run --locked --no-group cpu --group cu130 python train.py --config config/config_nerf_lego.yaml
```

Use the same group flags on subsequent `uv run` commands; omitting them selects
the CPU default again. The two accelerator groups are mutually exclusive and
both are recorded in `uv.lock`. Other dependencies come from PyPI. CUDA requires
a compatible NVIDIA driver; CUDA builds are unavailable on macOS. Training and
rendering use CUDA when PyTorch detects an available GPU, otherwise CPU.

Check the installed build and whether CUDA is usable:

```bash
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

For the CUDA install, add `--no-group cpu --group cu130` to that command too.
Checkpoints record the full PyTorch build, CUDA runtime, execution device, OS,
GPU names, memory and compute capabilities, and NVIDIA driver version when
available. An unavailable driver version is stored as `null`.

The locked CPU install and test suite were verified on 2026-09-27 with
PyTorch `2.14.0+cpu`, Python 3.12.3 and Linux x86-64 on an Intel i7-13700HX.
The CUDA install selection resolves to `2.14.0+cu130`. CUDA rendering was checked
on an RTX 4070 Laptop GPU (8 GB, compute capability 8.9) with driver 595.91.07;
GPU access required running outside the verification sandbox.

## How To Run?

### Training

Download the original NeRF example archive containing the synthetic `lego`
scene and the LLFF `fern` scene. Requires Bash, `wget`, and `unzip`.

```
bash download_dataset.sh
```

The script places the scenes in `datasets/lego/` and `datasets/fern/`, relative
to the script's location. Existing scene paths are preserved, and rerunning the
script downloads only when a scene is missing. Failed downloads, extraction,
or archive validation stop the script and remove its temporary files.

Choose `dataset_type: blender` for NeRF Synthetic scenes such as Lego, or
`dataset_type: llff` for forward-facing LLFF scenes such as Fern. Both formats
work with NeRF, SIREN and WaveletNeRF.

Configuration files use a flat YAML mapping. Start with an example in `config/`,
or use a minimal training config such as:

```yaml
experiment_name: nerf_lego_custom
model_type: nerf
dataset_type: blender
dataset_path: ./datasets/lego
learning_rate: 5e-4
num_random_rays: 1024
white_background: true
scene_center: [0, 0, 0]
scene_scale: 1
```

Omitted settings receive model/dataset defaults. Training requires a nonempty
`experiment_name`; evaluation override files can contain only the fields being
changed. Numbers and booleans must be native YAML values, not quoted strings.
Use `true`/`false` for all boolean fields and a three-number list for
`scene_center`. Unknown keys, duplicate keys, malformed YAML, unsupported model
options, invalid ranges and inconsistent bounds fail before data loading or
output creation. Both commands use the same model factory and accept the legacy
`multiscalewavelet` alias as `wavelet`.

Old `key = value` text config files are no longer accepted. Migrate their entries
to `key: value` in a `.yaml` or `.yml` file, keeping existing experiment settings
when resuming. Old checkpoint configs stored as strings are converted to native
types automatically. New checkpoints store the resolved typed configuration.

Device and performance settings can be set in YAML:

```yaml
device: auto         # CUDA when available, otherwise CPU; also cpu, cuda, cuda:1
compile_model: false # Opt into torch.compile with true
num_workers: 0       # Training DataLoader workers
chunk_size: 8192     # Rays rendered per chunk
netchunk: 65536      # Points queried per network call, for every model
```

Both commands accept `--device`, `--compile` / `--no-compile` and `--netchunk`;
training also accepts `--num-workers`. CLI values override YAML. For example:

```bash
uv run --locked python train.py --config config/config_siren_lego.yaml --device cpu --no-compile --num-workers 0
uv run --locked --no-group cpu --group cu130 python eval.py --checkpoint models/nerf/nerf_050000.pth --device cuda:0 --compile
```

Runtime choices can change on resume and evaluation; a checkpoint's device and
compilation settings are not inherited. Explicit unavailable CUDA devices fail
with an error. Compilation is off by default and uses Inductor when enabled;
compiler errors propagate, so use `--no-compile` in unsupported environments.
Pinned DataLoader memory is enabled only for CUDA. Point-query chunking bounds
each network call, but training still retains activations needed for gradients.
The render loops do not clear the CUDA allocator cache each frame. A small
float32 CUDA benchmark on the GPU above compared 20 frames per setting after
warmup, alternating the order and synchronizing for timing. It rendered 32×32
images with ray chunks of 512, point chunks of 8,192 and 64 samples per ray
(plus 32 fine samples for NeRF). Networks used width 64, eight spatial layers
for NeRF and four layers for SIREN/Wavelet. Median times were:

| Model | Keep cache | Clear cache before each frame |
| --- | ---: | ---: |
| NeRF | 14.17 ms | 15.12 ms |
| SIREN | 5.02 ms | 5.69 ms |
| Wavelet | 12.34 ms | 13.39 ms |

Rendered values matched between settings. These timings describe this small
case, rather than full training throughput. MPS and mixed precision are not supported;
the models and volume integration retain float32 arithmetic.
Inductor rendering and backward passes have been checked for NeRF, SIREN and
Wavelet on both PyTorch `2.14.0+cpu` and `2.14.0+cu130` on the hardware above.

Train the **baseline NeRF** on `lego`:

```
uv run --locked python train.py --config config/config_nerf_lego.yaml
```

Train the **SIREN-NeRF** on `lego`:

```
uv run --locked python train.py --config config/config_siren_lego.yaml
```

Train the **MFN (WaveletNet) NeRF** on `lego`:

```
uv run --locked python train.py --config config/config_wavelet_lego.yaml
```

Train on **Fern** using one of the corresponding configs:

```bash
uv run --locked python train.py --config config/config_nerf_fern.yaml
uv run --locked python train.py --config config/config_siren_fern.yaml
uv run --locked python train.py --config config/config_wavelet_fern.yaml
```

LLFF scenes require `poses_bounds.npy` and RGB images in `images/`, or an
existing `images_<dataset_factor>/` directory. Image filenames are sorted
lexicographically to match the pose rows. `config_nerf_fern.yaml` follows the
[original factor-8 Fern quick start](https://github.com/bmild/nerf/blob/master/config_fern.txt),
including 1,024 rays and 64 additional fine samples. Its explicit training budget
is 200,000 updates. The SIREN and wavelet configs also use factor 8. To select
the factor-4 paper settings, use `config/config_nerf_fern_paper.yaml`; it has its
own experiment name and retains 4,096 rays and 128 additional fine samples.
Cached images are used at their existing resolution, otherwise Pillow
resizes originals in memory. Loading does not write dataset files or require
ImageMagick. Focal lengths are adjusted to the actual resized dimensions.

The loader converts LLFF camera axes, scales translations and depth bounds by
`1 / (minimum_bound * llff_bounds_scale)`, and recenters the average camera when
`llff_recenter: true`. The Fern configs use `llff_bounds_scale: 0.75` and
`llff_holdout: 8`: every eighth view starting at index zero is held out, with
the remaining views used for training. Validation and test use the same held-out
views, following the original LLFF protocol. SIREN and wavelet Fern configs are
experimental starting settings; their benchmark quality has not been verified.
Checkpoints record the source indices, frame filenames, ordering, holdout stride
and shared validation/test policy, so the held-out views are reproducible.
Validation is therefore not an independent test set; avoid tuning on these
views when reporting final benchmark results.

Fern uses the [reference LLFF spiral](https://github.com/bmild/nerf/blob/master/load_llff.py):
120 views over two rotations, with camera radii derived from the 90th percentile
of camera positions and focus depth derived from scene bounds. Checkpoints save
the spiral settings and render intrinsics. `eval.py --mode render` reconstructs
the trajectory from those values without loading the dataset or a dummy image.
`--mode test` renders the actual held-out camera poses against their images.

`modules.datasets.load_scene` returns a `SceneData` containing RGB images,
OpenGL camera-to-world poses, per-view 3×3 intrinsics, camera-depth bounds,
source indices and paths, explicit splits, the source-world-to-scene transform,
sampling bounds, and render poses/path settings. `.split(name)` selects a
`SceneSplit`; `.describe()` produces serializable camera and preprocessing
metadata. Novel rendering uses a Lego orbit or an LLFF spiral with saved render
intrinsics, so dataset images are unnecessary.

Images must be 8-bit RGB or RGBA PNG, or RGB JPEG. Grayscale, palette images,
16-bit PNG, and other encodings must be converted explicitly before loading.
Decoded pixels are float32 RGB in `[0, 1]` (channel values divided by 255),
without gamma linearization or color-profile conversion. Blender resizes floating
channels with area averaging before optional white alpha compositing; without a
white background it retains the stored RGB, following the reference loader.
LLFF uses Lanczos resizing and composites any alpha over the configured background.
All loaded views must share their final image dimensions. Intrinsics are stored
per view and scaled separately in X and Y to match the actual resized dimensions.
Blender frame paths may begin with `./` and may include their extension; omitted
extensions default to `.png`. Paths must remain inside the dataset directory.
Ray geometry can be generated without target images using `camera_rays` or
`CameraRayGenerator`; `compute_rays` additionally validates and pairs RGB targets.

Logs are saved in:

```
./logs/<experiment_name>/
```

Model checkpoints are saved in:

```
./models/<experiment_name>/<experiment_name>_<step>.pth
```

`save_path` sets the checkpoint root directory; logs use `log_root`. For example,
`save_path: ./checkpoints` and `experiment_name: nerf_lego` save weights under
`./checkpoints/nerf_lego/`. All example configs use scene-specific experiment
names. Choose a distinct name for each new experiment to keep its outputs separate.
Older configs using `save_root` remain supported as an alias for `save_path`.
If both keys specify different directories, configuration loading fails with an
error. Resumed runs can override the output directory using either key; saved
checkpoint configs record only `save_path`.

Resume training from a checkpoint, using the config for that experiment:

```bash
# Lego
uv run --locked python train.py --config config/config_nerf_lego.yaml \
  --resume ./models/nerf_lego/nerf_lego_050000.pth

# Fern
uv run --locked python train.py --config config/config_nerf_fern.yaml \
  --resume ./models/nerf_fern_quickstart/nerf_fern_quickstart_050000.pth
```

Use a checkpoint that exists and set `num_iters` above its completed-update
count to continue training. For SIREN or wavelet, substitute the corresponding
config and experiment name: Lego uses `siren_lego` or `wavelet_lego`; Fern uses
`siren_fern` or `wavelet_fern`.

New training runs refuse nonempty experiment directories. Use `--resume` to
continue an existing run, choose a new `experiment_name`, or use `--overwrite`
to replace existing checkpoints, TensorBoard events and run metadata.
`--resume` and `--overwrite` are mutually exclusive. Overwrite preserves files
that are not recognized run artifacts.

The log directory contains a readable `config.yaml` with resolved defaults and
an `experiment.json` with dataset, code and environment metadata. Resume writes
`config.resume-<step>.yaml` and `experiment.resume-<step>.json`, keeping the original
run records. Checkpoints and these artifacts are written to a temporary file in
the destination directory and replaced atomically after a successful write.

New checkpoints include the resolved config (including defaults and seed),
training/validation camera poses, intrinsics and split indices, scene transforms,
camera-depth bounds and render-path settings,
white background convention, Git revision and dirty status, dependency versions,
and the `uv.lock` hash. Resume rejects changes to model settings, ray bounds,
training sampling and learning-rate settings, or dataset cameras/splits. You can
change run length, output paths and logging intervals, or relocate the same dataset.
The saved scene transform takes precedence over config scene settings.

Render a new checkpoint without the original config or dataset images:

```bash
uv run --locked python eval.py --checkpoint ./models/<exp>/<exp>_050000.pth
```

For the baseline examples, render a Lego orbit or a Fern spiral:

```bash
uv run --locked python eval.py --mode render \
  --checkpoint ./models/nerf_lego/nerf_lego_050000.pth --output ./renders/nerf_lego_orbit
uv run --locked python eval.py --mode render \
  --checkpoint ./models/nerf_fern_quickstart/nerf_fern_quickstart_050000.pth --output ./renders/nerf_fern_spiral
```

Evaluation refuses nonempty output directories. Choose a fresh `--output`, or
add `--overwrite` to replace generated `frame_*.png`, `test_*.png` and metrics.
This removes stale frames when a later render has fewer views and preserves
unrelated files.

`num_render_poses` controls frame count. Lego's `render_orbit_elevation` (degrees)
and `render_orbit_radius` (world units) control its orbit. Fern's
`render_spiral_rotations`, `render_spiral_zrate` and `render_spiral_radius_scale`
control turns, vertical oscillation rate and the multiplier on the camera-derived
spiral radii. Defaults preserve the original paths. These controls are saved in
checkpoint metadata and may be overridden using an `eval.py --config` file;
they change novel render cameras, without changing training or test cameras.

`--config` can override rendering sample count, chunk size and pose count; it
rejects conflicting model settings or ray bounds. Legacy checkpoints still need
the original config and warn that compatibility cannot be checked. New checkpoints also save Python, NumPy, Torch and CUDA RNG states and the
committed pixel-sampler state. Interrupted forwards resume from the last
completed update's random/sampling state. Validation uses separate randomness.
For reproducible runs, set `seed: 42` (or another unsigned 32-bit integer) and
`deterministic: true` in YAML. This enables Torch's deterministic algorithms
and disables cuDNN benchmarking; unsupported deterministic operations fail
explicitly. The actual determinism/backend settings are recorded. Matching an
uninterrupted trajectory requires the same data, execution settings, software
and hardware; it is not guaranteed across devices or versions. Legacy
checkpoints without RNG/sampling state warn that exact resume is unavailable.
Dataset images are not embedded in checkpoints.

### Training and rendering memory

Training stores RGB images and per-camera matrices, and generates ray geometry
only for the selected image/pixel indices in each batch. It does not allocate
full-scene origin/direction arrays or a full-scene shuffle permutation. Camera
metadata descriptions also avoid copying RGB splits. A virtual 100-view,
800×800 scene test samples 1,024 rays with less than 2 MiB of temporary allocation;
the actual float32 RGB images alone still occupy about 732 MiB at that size.

For baseline `no_batching: true`, each update selects one image and samples its
pixels, honoring the configured initial center crop. Otherwise batches sample
uniformly across training views. Sampling is without replacement within a batch,
but batches are independent: global sampling no longer consumes a shuffled
whole-scene epoch as the original implementation did. Checkpoints record this
protocol and the compact committed sampler RNG state. Worker prefetch can build
future batches, but checkpoint progress advances only after a successful update.
`num_workers: 0` avoids worker/prefetch overhead; additional workers can increase
host memory, particularly on platforms that copy dataset images when spawning.

Use `dataset_factor` to downsample either dataset before ray generation. Image
dimensions, intrinsics and split indices remain explicit in checkpoint metadata;
evaluation uses the same resolution. Blender's `half_res` multiplies its factor
by two when enabled. Increasing the factor changes the experimental resolution,
so keep it consistent when comparing results.

Validation and evaluation generate camera rays in `chunk_size` blocks, transfer
only those blocks to the rendering device, and immediately copy completed RGB
chunks into a CPU image buffer. Both rendering paths also support
`render_nerf(..., output_device='cpu')` inside `torch.no_grad()` for explicit
inference output buffering. Keep input rays on CPU to avoid allocating a full
image's rays on the GPU.

During training, ray chunks and point-query chunks are concatenated before the
single backward pass. Their autograd graphs retain the activations needed for
the whole training batch. Reducing `chunk_size` or `netchunk` bounds individual
operations and temporary workspace, but does **not** necessarily bound total
retained activation memory. Reduce `num_random_rays`, sample counts or model
width/depth when the training batch itself exceeds available GPU memory.

### Reference NeRF baseline

New `model_type: nerf` runs use `baseline_version: reference`, matching the
[original view-dependent MLP](https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py):
eight spatial ReLU layers, a skip after layer index 4, separate density and
256-channel feature projections, and one view-dependent RGB hidden layer.
Both independent coarse and fine networks use Glorot uniform weights and zero
biases. Position and direction encodings use 10 and 4 frequency bands.

The [reference renderer](https://github.com/bmild/nerf/blob/master/run_nerf.py)
uses original geometry coordinates, unnormalized geometry rays and unit world
viewing directions. It applies sigmoid to RGB logits and ReLU to raw density,
corrects integration intervals by ray length, and uses independent midpoint-bin
jitter on each training ray. Detached inverse-CDF samples from the coarse
weights are merged with coarse depths before querying the separate fine network.
Training minimizes coarse RGB MSE plus fine RGB MSE; evaluation renders the fine
network with deterministic sampling. LLFF uses NDC geometry with world viewing
directions, while Blender uses world geometry without a scene normalization.

The Lego config follows the
[published Blender settings](https://github.com/bmild/nerf/blob/master/paper_configs/blender_config.txt):
full resolution, white background, 1,024 pixels from one random image per update,
64 coarse samples plus 128 additional fine samples (192 fine queries per ray),
and 500,000 updates. `num_samples_eval` counts coarse samples too. Learning rate
is `5e-4 * 0.1 ** (completed_updates / 500000)` without a floor. The optimizer
matches TensorFlow 1.15 Keras Adam's epsilon-hat placement (`epsilon = 1e-7`).
`precrop_iters`, `precrop_frac`, `perturb`, `raw_noise_std`, `lindisp`,
`num_importance` and `no_batching` expose the corresponding sampling controls.
The original quick-start Lego config instead used half resolution and 64 fine
samples; those are different experimental settings.

For `dataset_type: llff`, reference defaults instead follow the
[published Fern settings](https://github.com/bmild/nerf/blob/master/paper_configs/llff_config.txt):
factor 4, holdout stride 8, 4,096 rays sampled across training images, 64 coarse
plus 128 fine samples, density noise standard deviation 1 during training,
200,000 updates and a 250,000-update learning-rate decay time constant. Explicit
config values override these defaults; lower-resolution quick-start runs use
different settings from the published experiment.

This is an algorithmic PyTorch translation, not a claim of bit-for-bit agreement
with TensorFlow random streams or kernels, nor verified reproduction of paper
scores. Existing nine-layer Softplus checkpoints automatically retain the
`legacy` baseline and its renderer. They cannot be converted into reference
coarse/fine weights: train a new experiment for reference comparisons.

### Fern ray conventions

Forward-facing LLFF scenes use the
[original NeRF NDC workflow](https://github.com/bmild/nerf/blob/master/run_nerf.py).
The loader scales and recenters the cameras using LLFF depth metadata. Rays are
shifted to the projection near plane at `1` and projected into normalized device
coordinates (NDC). Rendering then samples the projected rays linearly with
`near: 0` and `far: 1`; these bounds are distinct from the projection plane
and the original world-space depth bounds.

Projected geometry directions retain their magnitude. Both renderers multiply
sample intervals by that magnitude during volume integration, and the appearance
head receives the original unit world-space viewing directions. Training uses
independent midpoint-bin jitter for each ray; validation and evaluation use
deterministic samples. NDC positions pass to every model without an additional
scene transform (`scene_center: [0, 0, 0]`, `scene_scale: 1`). LLFF configurations
reject incompatible ray bounds, inverse-depth sampling and white backgrounds.
Lego's `2–6` bounds and scene scale must not be carried over to Fern.

These conventions change the geometry learned by the model. Train fresh Fern
experiments; existing weights trained with world rays or different bounds are
not converted into NDC weights.

### More Datasets

Scene coordinates are configured independently of ray sampling bounds. Set
`scene_center: [x, y, z]` and a positive `scene_scale` (the half-extent of a
world-space cube). SIREN, Wavelet and legacy NeRF networks receive
`(position - scene_center) / scene_scale`. The SIREN and Wavelet Lego configs use
center `(0, 0, 0)` and scale `2`, mapping the cube
`[-2, 2]³` to `[-1, 1]³`. This is an explicit scene convention, not a fitted
object bounding box. Points outside it are not clipped. Without these settings,
new runs use center `(0, 0, 0)` and scale `1`. Choose appropriate settings for
other scenes. `near` and `far` only control sampling distances along world-space
rays; ray directions, integration intervals, and density units are unchanged.

Every training checkpoint saves its scene transform. Resume and evaluation use
that saved transform even if the config's scene settings have changed. Older
checkpoints without this metadata retain the previous near/far-based coordinate
mapping with a warning; supply their original `near` and `far` values. Resaving
them records that legacy transform explicitly. New normalization changes the
coordinates learned by the networks, so retrain experiments before comparing
results; resuming old weights does not convert them to the new convention.

To use other scenes from the **NeRF Synthetic dataset**, you can download all datasets from:

[https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset)

Unzip them and place the scene folders inside the `datasets/` directory of this repository.  
The structure should look like this:

```
datasets/
├── lego/
├── chair/
├── drums/
├── ficus/
├── hotdog/
├── materials/
├── mic/
└── ship/
```

To train on a different dataset, edit the `dataset_path` parameter in the config file.  
For example, to train on **chair**, set:

```
dataset_path: ./datasets/chair
```

Then run:

```
uv run --locked python train.py --config config/config_nerf_lego.yaml
```

This example uses the existing baseline config after changing its dataset path.
Also choose a new `experiment_name` so its outputs are separate from Lego runs.

### Evaluate test views

Render every Blender test camera or every held-out LLFF camera and compare it
against its image:

```bash
uv run --locked python eval.py --mode test \
  --checkpoint ./models/nerf_lego/nerf_lego_250000.pth \
  --dataset-path ./datasets/lego \
  --output ./renders/nerf_lego_test

uv run --locked python eval.py --mode test \
  --checkpoint ./models/nerf_fern_quickstart/nerf_fern_quickstart_050000.pth \
  --dataset-path ./datasets/fern \
  --output ./renders/nerf_fern_test
```

New checkpoints restore their model settings automatically. For legacy checkpoints,
also supply `--config config/config_nerf_lego.yaml`. Test evaluation requires the
dataset images; `--dataset-path` can relocate the dataset without changing model
settings. Blender test frames follow JSON order; LLFF test frames follow the
sorted image order and configured holdout interval. All use their actual poses,
resolution and per-view intrinsics. Blender `testskip` affects training validation
subsampling; test evaluation always scores the full test split. Uniform ray
sampling makes evaluation deterministic.

The output contains predicted `test_0000.png`, etc., `metrics.csv` with per-image
MSE/PSNR, and `metrics.json` with per-image values, aggregate results and rendering
settings. `mean_psnr_db` is the arithmetic mean of per-image PSNR; `pooled_psnr_db`
is PSNR computed from the pixel-weighted mean squared error across all images.
These are different statistics. PSNR uses a peak value of 1, full RGB images,
float64 squared-error accumulation, no mask/crop and no linear-light conversion.
Blender RGBA targets are composited over white when `white_background: true`;
otherwise the stored RGB channels are retained, following the reference Blender
loader. Fern uses black-background rendering with RGB targets. Metrics use float
predictions before PNG clipping or eight-bit quantization. Exact matches have
infinite PSNR, represented as the string `"Infinity"` in standard JSON and `inf`
in CSV. SSIM and LPIPS are not currently reported; PSNR alone should not be
presented as a complete reproduction of a multi-metric published benchmark.

### Verify the downloaded examples

Run a short CPU check on the real downloaded data for all six combinations of
Lego/Fern and NeRF/SIREN/wavelet:

```bash
bash download_dataset.sh
uv run --locked python scripts/verify_examples.py
```

This uses each example's model architecture and scene conventions, but reduces
resolution to factor 32, training to one update followed by a resume to two,
sampling to four coarse samples (plus four fine samples for NeRF), and novel
rendering to two frames. It runs validation, evaluates every test view, checks
finite MSE/PSNR and output counts, and saves configs, logs, checkpoints, metrics,
frames and `verification.json` in a fresh directory under
`renders/example_verification/`. Generated outputs and downloaded data are
ignored by Git. `--factor` and `--output` can change verification resolution
and destination; the example configs themselves are never modified.

This check passed on the official archive on 2026-09-26 for all six cases:
200 Lego test images per model at 25×25 and three held-out Fern images per model
at 94×126. The downloader installed both scenes, retained Fern's cached factor-4
and factor-8 images, and preserved existing scene paths on rerun. This verifies
the CPU execution workflow on actual data. Full training at the example
resolutions, GPU runs on the downloaded scenes and reproduction of benchmark scores have not been
verified by this check.

### Render a video

Once you have trained a model, render frames with:
```
uv run --locked python eval.py \
  --mode render \
  --config config/config_nerf_lego.yaml \
  --checkpoint ./models/nerf_lego/nerf_lego_250000.pth \
  --output ./renders/nerf_lego_eval
```

Then you can make a video with this ffmpeg command:
```
ffmpeg -y -framerate 30 -i ./renders/nerf_lego_eval/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 ./renders/nerf_lego_eval.mp4
```

FFmpeg is an external command required only for this video conversion step.
Inspect training logs with:

```bash
uv run --locked tensorboard --logdir logs
```

## Development

Runtime dependencies cover tensor computation, NumPy/image handling, progress,
TensorBoard and YAML configuration. Ruff is a development dependency. The former
Conda-only extras (`torchvision`, `torchaudio`, notebook tools, Matplotlib and
scikit-learn) and the `mkl<2024.1` pin are not retained.

The PyTorch minimum is 2.4. PyTorch 2.2 CPU failed compilation on Python 3.12
and could not exchange arrays with the locked NumPy 2.5.3. PyTorch 2.4 CPU was
tested with Python 3.12.3 and NumPy 2.5.3, including compilation, `weights_only`
checkpoint loading and the test suite. CI checks this minimum separately from
the locked environment. Other dependency lower bounds are not a separately
tested minimum-version combination; use the committed lockfile for reproducible
runs.

Run the dependency, lint, and command-line checks used in CI:

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked python train.py --help
uv run --locked python eval.py --help
uv run --locked python -m unittest discover -s tests
```

Use `uv add <dependency>` or `uv add --dev <tool>` when changing dependencies,
and commit both `pyproject.toml` and `uv.lock`. To update an existing dependency
deliberately, run `uv lock --upgrade-package <dependency>`, then
`uv sync --locked` and the checks above. No distribution build is required.

## Method

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution  
  
![NeRF pipeline figure from Mildenhall et al.](imgs/pipeline.jpg)

Pipeline figure from Mildenhall et al., *NeRF: Representing Scenes as Neural
Radiance Fields for View Synthesis* (ECCV 2020), obtained from the
[original NeRF repository](https://github.com/bmild/nerf/blob/master/imgs/pipeline.jpg).

> A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views


## Citation

We acknowledge the original authors of NeRF for their groundbreaking work:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License and attribution

Project-authored code and documentation, and the combined software including
the MFN adaptation, are licensed under [AGPL-3.0-only](LICENSE). Upstream MIT
notices for NeRF and SIREN are preserved in the same file.

The MFN base and Gabor-style filter construction in `modules/models.py` are
adapted from [Fathony et al.'s implementation](https://github.com/boschresearch/multiplicative-filter-networks).
Frequency controls, optional LayerNorm, and the NeRF integration are local
extensions. The SIREN layers draw on
[Sitzmann et al.'s implementation](https://github.com/vsitzmann/siren).

The pipeline figure belongs to the original NeRF authors and is not relicensed
by this project. The Lego GIF was generated with this repository; its
project-authored contributions are offered under AGPL-3.0-only, without changing
rights in the underlying scene assets.

The downloader uses the example archive distributed by the
[original NeRF project](https://github.com/bmild/nerf), which contains Lego and
Fern. The script retains both scenes. Downloaded datasets remain subject
to their original rights-holder terms; this repository's software license
grants no additional dataset rights. See [LICENSE](LICENSE) for source links,
attribution, and the complete notices.
