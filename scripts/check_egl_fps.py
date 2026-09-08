"""Phase 0 check 5: prove MuJoCo renders on the GPU via EGL, not on the CPU.

A broken EGL chain does not raise. MuJoCo silently falls back to Mesa's
`llvmpipe` software rasteriser and gets 10-50x slower, which would only show up
weeks later as "why is my eval so slow". So we check two independent things:

  1. GL_RENDERER  - the decisive evidence. Says which driver owns the context.
  2. render FPS   - the consequence. Hundreds = GPU, tens = llvmpipe.

Render size is 256x256 to match the LIBERO camera resolution we will actually use.
"""

import ctypes
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set BEFORE importing mujoco

import mujoco  # noqa: E402

WIDTH = HEIGHT = 256
N_FRAMES = 1000

XML = """
<mujoco>
  <visual><global offwidth="640" offheight="640"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.2 0.2 1"/>
    </body>
    <body pos="0.4 0.2 0.6">
      <freejoint/>
      <geom type="sphere" size="0.08" rgba="0.2 0.4 0.9 1"/>
    </body>
    <camera name="cam" pos="1.4 -1.4 1.0" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""


def gl_strings() -> dict:
    """Query the live GL context. Must run after a Renderer exists."""
    out = {}
    for libname in ("libGL.so.1", "libGLESv2.so.2"):
        try:
            gl = ctypes.CDLL(libname)
            gl.glGetString.restype = ctypes.c_char_p
            gl.glGetString.argtypes = [ctypes.c_uint]
            got = {}
            for name, enum in (("VENDOR", 0x1F00), ("RENDERER", 0x1F01),
                               ("VERSION", 0x1F02)):
                val = gl.glGetString(enum)
                got[name] = val.decode() if val else None
            if got["RENDERER"]:
                out = got
                out["_via"] = libname
                break
        except OSError:
            continue
    return out


print(f"MUJOCO_GL = {os.environ['MUJOCO_GL']}")
print(f"mujoco    = {mujoco.__version__}")

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

info = gl_strings()
print("\n-- GL context (the decisive evidence) --")
for k in ("VENDOR", "RENDERER", "VERSION"):
    print(f"  GL_{k:<9} = {info.get(k)}")

renderer_name = (info.get("RENDERER") or "").lower()
software = any(s in renderer_name for s in ("llvmpipe", "softpipe", "swrast"))
print(f"\n  software rasteriser detected = {software}")

# warmup - first frames pay context/shader setup costs
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="cam")
for _ in range(20):
    renderer.render()

print(f"\n-- rendering {N_FRAMES} frames at {WIDTH}x{HEIGHT} --")
t0 = time.perf_counter()
for _ in range(N_FRAMES):
    px = renderer.render()
elapsed = time.perf_counter() - t0

fps = N_FRAMES / elapsed
print(f"  elapsed      = {elapsed:.3f} s")
print(f"  render FPS   = {fps:.1f}")
print(f"  ms per frame = {elapsed / N_FRAMES * 1000:.3f}")
print(f"  frame shape  = {px.shape}, dtype={px.dtype}, "
      f"mean={px.mean():.2f} (0.00 would mean a blank image)")

# physics + render, closer to what a rollout actually costs
mujoco.mj_resetData(model, data)
t0 = time.perf_counter()
for _ in range(N_FRAMES):
    mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="cam")
    renderer.render()
step_elapsed = time.perf_counter() - t0
print(f"\n  step+render FPS = {N_FRAMES / step_elapsed:.1f} "
      f"({step_elapsed:.3f} s for {N_FRAMES} iters)")

print(f"\nVERDICT: {'FAIL - fell back to CPU software rendering' if software else 'PASS - GPU rendering via EGL'}")
