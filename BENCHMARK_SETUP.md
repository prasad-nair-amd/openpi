# pi0 Benchmark Setup on AMD ROCm

Step-by-step guide to run `benchmark_pi0_rocm.py` on AMD Instinct GPUs.

**Tested environment**
- OS: Ubuntu 24.04.3 LTS
- GPUs: 8× AMD Instinct MI300X
- ROCm: 7.8.0 (rocm-smi 4.0.0)
- Python: 3.11.14 (via uv venv)
- PyTorch: 2.11.0+rocm7.2

---

## 1. Prerequisites

### 1.1 ROCm system stack

Verify the ROCm kernel driver and `rocm-smi` are installed:

```bash
rocm-smi
```

Expected: a table listing your AMD GPUs with temperature, power, and VRAM usage.
If this fails, install ROCm from the [AMD ROCm documentation](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html).

Also confirm device files exist:

```bash
ls /dev/kfd /dev/dri/renderD*
```

### 1.2 Git LFS

openpi uses Git LFS for large model assets:

```bash
git lfs install
```

### 1.3 uv

openpi uses `uv` as its package manager:

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

---

## 2. Clone and install openpi

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Skip LFS download of large binaries during clone
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

This creates a `.venv` inside the repo with Python 3.11 and all dependencies.

> **Why `GIT_LFS_SKIP_SMUDGE=1`?**  
> Prevents uv from trying to download multi-GB model files during env setup.
> Checkpoints are downloaded separately on demand.

---

## 3. Install PyTorch with ROCm support

The default `uv sync` installs a CUDA build of PyTorch which cannot use AMD GPUs.
Replace it with the ROCm wheel **directly into the uv venv**, bypassing the lockfile:

```bash
.venv/bin/pip install torch==2.7.1 torchvision \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  --force-reinstall
```

> **Do not use `uv run pip install` or `uv pip install` here.**  
> `uv run` re-syncs from `uv.lock` on every invocation and will overwrite the ROCm
> wheel with the locked CUDA build. Using `.venv/bin/pip` directly bypasses the lock.

Verify:

```bash
.venv/bin/python -c "
import torch
print('version:', torch.__version__)        # expect 2.x.x+rocm7.2
print('HIP:    ', torch.version.hip)        # expect 7.2.x
print('GPUs:   ', torch.cuda.device_count()) # expect > 0
"
```

Expected output:

```
version: 2.x.x+rocm7.2
HIP:     7.2.xxxxx
GPUs:    8
```

### Multiple Python environments — what goes where

This repo has two Python environments that coexist:

| Environment | Path | Python | Purpose |
|---|---|---|---|
| conda `base` | `/home/amd/miniconda3/` | 3.13 | System default, not used for openpi |
| uv project venv | `openpi/.venv/` | 3.11 | openpi runtime — use this |

Always use `.venv/bin/python` (or activate with `source .venv/bin/activate`) to
ensure you are using the ROCm-enabled torch.

---

## 4. Download and prepare the checkpoint

### 4.1 Check for a cached checkpoint

```bash
ls ~/.cache/openpi/openpi-assets/checkpoints/
```

The `pi0_base` checkpoint (base weights used by all fine-tuned configs) may already
be present from a prior download.

### 4.2 Download if not cached

The benchmark script downloads automatically from GCS if `--checkpoint-dir` is not
provided, but this requires `gsutil` or GCS credentials.

Download manually using `gcsfs` (already installed as a dependency):

```bash
.venv/bin/python -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
fs.get(
    'openpi-assets/checkpoints/pi0_base',
    str(Path.home() / '.cache/openpi/openpi-assets/checkpoints/pi0_base'),
    recursive=True,
)
print('done')
"
```

Or with gsutil if available:

```bash
gsutil -m cp -r \
  gs://openpi-assets/checkpoints/pi0_base \
  ~/.cache/openpi/openpi-assets/checkpoints/pi0_base
```

### 4.3 Convert JAX checkpoint to PyTorch

The downloaded checkpoint is in JAX/Orbax format. Convert it to `model.safetensors`
for the PyTorch backend:

```bash
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base \
  --config_name pi0_aloha \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch
```

### 4.4 Copy normalization statistics

The conversion script writes only `model.safetensors`. The policy also needs
per-robot normalization stats, which live in the original `assets/` directory:

```bash
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi0_base/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch/assets
```

Verify the final checkpoint layout:

```bash
find ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch -maxdepth 3
```

Expected:

```
pi0_base_pytorch/
├── model.safetensors          ← PyTorch weights
└── assets/
    ├── trossen/
    │   └── norm_stats.json    ← required for pi0_aloha
    ├── droid/
    │   └── norm_stats.json
    └── ...
```

---

## 5. Run the benchmark

### Quick smoke test (3 inference passes)

```bash
.venv/bin/python benchmark_pi0_rocm.py \
  --config pi0_aloha \
  --backend pytorch \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  --quick
```

### Full benchmark (20 timed runs)

```bash
.venv/bin/python benchmark_pi0_rocm.py \
  --config pi0_aloha \
  --backend pytorch \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
  --num-warmup 5 \
  --num-runs 20 \
  --output-json results/pi0_aloha_rocm.json
```

### All CLI options

```
--config          Model config name (default: pi0_aloha)
                  Choices: pi0_aloha, pi05_aloha, pi0_droid, pi0_fast_droid,
                           pi05_droid, pi0_libero, pi0_fast_libero, pi05_libero, ...

--backend         pytorch  (ROCm torch — recommended)
                  jax      (requires jax[rocm] wheel — see section 6)

--checkpoint-dir  Path to local checkpoint directory (skips GCS download)

--num-warmup      Warmup passes before timing starts (default: 3)
--num-runs        Number of timed inference passes (default: 20)
--image-size H W  Synthetic image resolution (default: 224 224)
--output-json     Write JSON results to this path
--quick           Shorthand: 1 warmup + 3 runs
--list-devices    Print detected AMD GPUs and exit
```

---

## 6. JAX ROCm backend (optional)

The JAX backend requires a separate ROCm-enabled JAX wheel. It is not installed
by `uv sync` by default.

```bash
.venv/bin/pip install jax[rocm] \
  -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
```

Verify:

```bash
.venv/bin/python -c "import jax; print(jax.devices())"
# expect: [RocmDevice(id=0), ...]
```

Then run:

```bash
.venv/bin/python benchmark_pi0_rocm.py \
  --config pi0_aloha \
  --backend jax \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base \
  --quick
```

> The JAX backend uses the original JAX/Orbax checkpoint directly — no conversion needed.

---

## 7. Troubleshooting

### `No AMD/ROCm devices detected`

The benchmark uses `rocm-smi` as the primary detection method. If it still shows
no devices after `rocm-smi` works in the shell, you are likely running with the
wrong Python (conda base instead of `.venv`). Always prefix with `.venv/bin/python`.

### `torch.cuda.is_available()` returns `False`

Your torch wheel is a CUDA build, not ROCm. Check:

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
# Must end in +rocmX.Y, not +cuXXX
```

If it shows `+cuXXX`, re-run step 3.

### `uv run` reinstalls CUDA torch

`uv run` syncs against `uv.lock` which pins the CUDA build. Use `.venv/bin/python`
directly instead of `uv run python`.

### `Norm stats file not found`

The `assets/` folder was not copied after conversion. Re-run step 4.4.

### `KeyError: 'state'` or `KeyError: 'images'`

The observation format must match what each policy expects exactly. The benchmark
uses the `make_*_example()` functions from each policy module as the reference.
This is already handled in `benchmark_pi0_rocm.py` — ensure you are using the
latest version of the script.

### GCS download fails (`gsutil not found`)

Either install `gsutil` via `pip install gsutil` and configure GCP credentials,
or use the `gcsfs` manual download shown in step 4.2. The `pi0_base` checkpoint
is publicly accessible without authentication.

---

## 8. Checkpoint and config reference

| Config | Base weights | Robot | State dim | Cameras |
|---|---|---|---|---|
| `pi0_aloha` | `pi0_base` | Trossen dual-arm | 14 | cam_high, cam_low, cam_left_wrist, cam_right_wrist |
| `pi05_aloha` | `pi0_base` | Trossen dual-arm | 14 | same as above |
| `pi0_droid` | `pi0_base` | DROID | 7 joints + 1 gripper | exterior_image_1_left, wrist_image_left |
| `pi0_libero` | `pi0_base` | LIBERO tabletop | 8 | image, wrist_image |

All configs load weights from `pi0_base`. A single converted checkpoint covers all
of them — only the `assets/<robot>/norm_stats.json` differs per config.
