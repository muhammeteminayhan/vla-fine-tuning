# Phase 0 — Environment Setup and Verification

**Date:** 2026-09-08
**Gate criterion:** all six checks pass, filled in with real command output.
**Status:** ✅ 6/6 passed.

Every number on this page comes from `results/phase0_checks.log`, produced by:

```bash
conda activate lerobot
bash scripts/run_phase0_checks.sh > results/phase0_checks.log 2>&1
```

The purpose of this phase is narrow but important: make sure that when something
breaks in Phase 3 or 4, "the environment is broken" is not on the list of
suspects.

---

## 1. Hardware and OS

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 **Laptop** GPU |
| Compute capability | **12.0** (`sm_120`, Blackwell) |
| VRAM (nominal) | 8151 MiB |
| VRAM (actually free) | **7.359 GiB** — see check 4 |
| Driver | 595.84 (open kernel module) |
| CUDA supported by driver | 13.2 |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic |
| CPU RAM | 31 GiB |
| Free disk | 215 GiB |

---

## 2. Software stack

| Package | Version | Notes |
|---|---|---|
| Python | 3.12.14 | conda env `lerobot`, Miniforge at `~/miniforge3` |
| PyTorch | **2.11.0+cu130** | LeRobot pins `torch>=2.7,<2.12.0`; 2.11 is the ceiling |
| torchvision | 0.26.0 | |
| torchcodec | 0.11.1 | video decoding backend |
| lerobot | 0.6.2 | source install, `~/lerobot` |
| hf-libero | 0.1.4 | LIBERO now ships as a PyPI package |
| robosuite | 1.4.0 | |
| robomimic | 0.2.0 | |
| bddl | 1.0.1 | task definition language |
| mujoco | 3.8.1 | |
| peft | 0.20.0 | |
| transformers | 5.5.4 | |
| gymnasium | 1.3.0 | |
| accelerate | 1.14.0 | |
| numpy | 2.2.6 | |
| wandb | 0.27.2 | |
| ffmpeg | **7.1.1** | pinned down from 9.0.1 — see "Problems hit" below |

**LeRobot source commit:** `2774d9bddcbbda50e697e162e89e7eaada8d7105` (2026-09-07)

Extras installed: `.[libero,peft,smolvla,training]`

---

## 3. The six checks

### Check 1 — `nvidia-smi`

Expected: card visible, driver 580+.

```
name, driver_version, compute_cap, memory.total [MiB]
NVIDIA GeForce RTX 5060 Laptop GPU, 595.84, 12.0, 8151 MiB
```

**PASS.** Driver 595.84 is well above the 580 floor.

---

### Check 2 — open kernel module

Expected: `Dual MIT/GPL`.

```
version:        595.84
license:        Dual MIT/GPL
```

**PASS.** The open kernel module is in use, which is required for Blackwell.

---

### Check 3 — torch, CUDA, compute capability

Expected: `True`, `(12, 0)`.

```
torch.__version__            = 2.11.0+cu130
torch.version.cuda           = 13.0
torch.cuda.is_available()    = True
torch.cuda.get_device_name() = NVIDIA GeForce RTX 5060 Laptop GPU
get_device_capability()      = (12, 0)
torch.cuda.get_arch_list()   = ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
this GPU needs               = sm_120
sm_120 in arch_list           = True

-- real GPU matmul (this is what actually proves sm_120 works) --
fp32 matmul 4096x4096 -> (4096, 4096), sum=-529245.7500
bf16 matmul 4096x4096 -> (4096, 4096), dtype=torch.bfloat16
```

**PASS.** Note that the check does not stop at `sm_120 in arch_list`. Metadata
can claim support that the shipped kernels do not deliver, so the check launches
a real fp32 and bf16 matmul. Both complete.

---

### Check 4 — free VRAM (the training budget)

```
mem_get_info() free  =  7,901,282,304 B = 7.359 GiB
mem_get_info() total =  8,082,096,128 B = 7.527 GiB
used by others       = 172.4 MiB
```

**PASS, with a correction to our own assumption.** The project brief said
"8 GB VRAM". The real, usable figure is **7.359 GiB** — the card reports 7.527 GiB
total and Xorg plus the desktop already hold 172 MiB. Phase 3's batch-size search
budgets against 7.36 GiB, not 8.

---

### Check 5 — MuJoCo EGL render benchmark

This is the check most likely to fail silently. When the EGL chain is broken,
MuJoCo does not raise: it falls back to Mesa's `llvmpipe` software rasteriser and
runs 10-50x slower. So the check measures two independent things — which driver
owns the GL context, and what the resulting throughput is.

```
MUJOCO_GL = egl
mujoco    = 3.8.1

-- GL context (the decisive evidence) --
  GL_VENDOR    = NVIDIA Corporation
  GL_RENDERER  = NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2
  GL_VERSION   = 4.6.0 NVIDIA 595.84

  software rasteriser detected = False

-- rendering 1000 frames at 256x256 --
  elapsed      = 0.181 s
  render FPS   = 5539.1
  ms per frame = 0.181
  frame shape  = (256, 256, 3), dtype=uint8, mean=122.39 (0.00 would mean a blank image)

  step+render FPS = 4999.1 (0.200 s for 1000 iters)

VERDICT: PASS - GPU rendering via EGL
```

**PASS, decisively.** `GL_RENDERER` names the actual GPU, not `llvmpipe`. At
**5539 FPS** for 256×256 frames the gate ("hundreds of FPS") is cleared by an
order of magnitude. The `mean=122.39` line guards against the other silent
failure mode: a context that renders successfully but produces black frames.

Render resolution is 256×256 because that is the LIBERO camera resolution this
project actually uses.

---

### Check 6 — one LIBERO episode end to end

Random actions, so success is not expected. What is being tested is that the
whole chain — bddl task files → robosuite → MuJoCo → EGL → pixels → mp4 — runs
start to finish without error.

```json
{
  "suite": "libero_object",
  "task_id": 0,
  "task": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
  "seed": 42,
  "steps": 300,
  "terminated": false,
  "truncated": false,
  "success": false,
  "frames_recorded": 301,
  "frame_shape": [256, 256, 3],
  "wall_clock_s": 3.065,
  "env_steps_per_s": 97.9,
  "sim_speed_vs_realtime": 4.89,
  "video": "assets/phase0_libero_episode.mp4",
  "video_bytes": 158143
}
```

**PASS.** Video written: `assets/phase0_libero_episode.mp4`.

`env_steps_per_s = 97.9` is a number Phase 4 will need. At 4.89x realtime, the
simulator itself is not going to be the bottleneck in evaluation — policy
inference will be.

---

## 4. Facts discovered here that later phases depend on

- **`fps = 20`** in `LiberoEnv` config, matching robosuite's `control_freq`.
  Phase 2's operator-minute model converts episode length to seconds by dividing
  frame count by this number, so it is a load-bearing constant, not trivia.
- **Observations are 256×256**, two cameras (`image` = agentview,
  `image2` = wrist), state is 8-dim, actions are `Box(-1, 1, (7,))`.
- **`control_mode` defaults to `relative`.** It must match how the policy
  checkpoint was trained; a mismatch shows up as quietly poor success rates
  rather than an error. Fix it once and log it on every run.
- **`hard_reset=False` requires `init_states=True`** — enforced in
  `LiberoEnv.__post_init__`. Soft resets are faster but not bit-identical, so
  Phase 1's determinism gate must stay on hard resets.
- **`info` carries `is_success`, `task`, `task_id`, `done`** — this is where the
  Phase 1 harness reads its success signal from.
- `lerobot/libero` has 40 tasks (4 suites × 10) over 1693 episodes, so roughly
  **42 episodes per task**. Phase 2's `K=40` upper end sits right at that ceiling
  and needs checking against the real per-task distribution.

---

## 5. Problems hit, and how they were resolved

Recorded because reproducing this environment means hitting them again.

### 5.1 `git`, `gcc`, `make`, `conda`, `ffmpeg` all missing

The machine was bare. `sudo` requires a password and could not be used
non-interactively, so everything was installed through Miniforge and conda-forge
instead of `apt`. This turned out to be an advantage: the whole setup now needs
no root at all.

### 5.2 `egl_probe` failed to build — CMake 4 dropped old policy support

Dependency chain: `hf-libero` → `robomimic==0.2.0` → `egl_probe`.

```
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
  Or, add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.
subprocess.CalledProcessError: Command 'cmake ..; make -j' returned non-zero exit status 2.
```

`egl_probe`'s `CMakeLists.txt` predates CMake 4. Resolved with CMake's own
documented escape hatch:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
```

Note that `hf-libero` already ships the maintained fork (`hf-egl-probe 1.0.2`);
the old package is only pulled in because `robomimic 0.2.0` still pins it.

### 5.3 torchcodec could not load against ffmpeg 9.0.1

conda-forge installed ffmpeg 9.0.1. torchcodec 0.11.1 only ships
`libtorchcodec_core4` through `core7`:

```
OSError: Could not load this library:
  .../site-packages/torchcodec/libtorchcodec_core4.so
```

Resolved with the pin the LeRobot docs already recommend for exactly this case:

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

### 5.4 LIBERO prompts interactively on first import

`libero/libero/__init__.py` calls `input()` the first time it runs, which fails
with `EOFError` in any non-interactive context. It only does this when
`~/.libero/config.yaml` is absent, so creating that file once is enough:

```bash
echo "N" | python -c "import libero.libero"
```

LIBERO assets (586 files) then download automatically to `~/.cache/libero/assets`.

---

## 6. Reproducing this environment

```bash
# 1. Miniforge (no root required)
curl -fsSL -o Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# 2. Environment and build toolchain
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge git "ffmpeg=7.1.1" cmake make c-compiler cxx-compiler pkg-config

# 3. PyTorch FIRST and ALONE, then verify sm_120 before installing anything else
pip install "torch==2.11.*" torchvision
python scripts/check_gpu.py          # must pass before continuing

# 4. LeRobot, with torch pinned so no transitive dep can swap the wheel
printf 'torch==2.11.0\ntorchvision==0.26.0\n' > torch-constraints.txt
git clone https://github.com/huggingface/lerobot.git "$HOME/lerobot"
cd "$HOME/lerobot" && git checkout 2774d9bddcbbda50e697e162e89e7eaada8d7105
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # for egl_probe, see 5.2
pip install -c /path/to/torch-constraints.txt -e ".[libero,peft,smolvla,training]"

# 5. LIBERO first-run config (avoids the interactive prompt, see 5.4)
echo "N" | python -c "import libero.libero"

# 6. Verify
export MUJOCO_GL=egl
bash scripts/run_phase0_checks.sh
```

---

## 7. Gate decision

| # | Check | Result |
|---|---|---|
| 1 | `nvidia-smi` | ✅ driver 595.84 |
| 2 | open kernel module | ✅ `Dual MIT/GPL` |
| 3 | torch / sm_120 | ✅ `(12, 0)`, real matmul runs |
| 4 | free VRAM | ✅ 7.359 GiB measured |
| 5 | MuJoCo EGL FPS | ✅ 5539 FPS, `GL_RENDERER` = NVIDIA |
| 6 | LIBERO episode | ✅ 300 steps, video written |

**Phase 0 gate: PASSED.**
