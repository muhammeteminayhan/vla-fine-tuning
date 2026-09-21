"""Guards for the assumptions the reported numbers rest on.

These are not unit tests for their own sake. Each one protects a property that,
if it silently broke, would leave the study producing plausible-looking numbers
that mean something other than what the report claims:

  * Wilson intervals behave correctly at 0 and 1, which is where this project
    lives and where the normal approximation fails.
  * K-shot subsets stay deterministic and nested, so the curve measures how many
    demonstrations were given rather than which ones.
  * The held-out suite never leaks into stage-1 training.
  * The seed recorded per episode matches the formula the harness reconstructs
    it with.

Run with: pytest tests -q          (no GPU, no dataset download)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from common.stats import wilson  # noqa: E402

# --- Wilson interval ------------------------------------------------------

def test_wilson_never_leaves_the_unit_interval():
    for n in (1, 6, 18, 50, 150, 1000):
        for k in (0, 1, n // 2, n - 1, n):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0, (k, n, lo, hi)


def test_wilson_is_not_degenerate_at_zero_successes():
    """The whole reason for using Wilson: 0/n does not mean 'exactly 0%'."""
    lo, hi = wilson(0, 6)
    assert lo == 0.0
    assert 0.35 < hi < 0.45, hi          # ~39% - a normal approximation gives 0
    lo18, hi18 = wilson(0, 18)
    assert hi18 < hi, "more trials must tighten the bound"


def test_wilson_tightens_with_sample_size():
    widths = [wilson(n // 2, n)[1] - wilson(n // 2, n)[0] for n in (10, 50, 150, 600)]
    assert widths == sorted(widths, reverse=True)


def test_wilson_matches_reported_numbers():
    """The intervals printed in docs/04-findings.md must be reproducible."""
    lo, hi = wilson(98, 150)                     # K=5, three seeds
    assert round(lo * 100, 1) == 57.4
    assert round(hi * 100, 1) == 72.5


# --- K-shot splits --------------------------------------------------------

@pytest.fixture(scope="module")
def splits():
    return json.loads((REPO / "configs" / "kshot_splits.json").read_text())


def test_subsets_are_nested(splits):
    ks = sorted(splits["k_values"])
    for seed in splits["seeds"]:
        for small, large in zip(ks, ks[1:], strict=False):
            a = set(splits["splits"][f"k{small}_seed{seed}"]["episodes"])
            b = set(splits["splits"][f"k{large}_seed{seed}"]["episodes"])
            assert a < b, f"k{small} is not a strict subset of k{large} at seed {seed}"


def test_different_seeds_give_different_subsets(splits):
    a = set(splits["splits"]["k5_seed0"]["episodes"])
    b = set(splits["splits"]["k5_seed1"]["episodes"])
    assert a != b
    assert len(a & b) < len(a), "seeds must not draw the same subset"


def test_every_split_has_k_episodes_for_each_task(splits):
    for name, spec in splits["splits"].items():
        for task, eps in spec["episodes_per_task"].items():
            assert len(eps) == spec["K"], f"{name} task {task}"
            assert len(set(eps)) == len(eps), f"{name} task {task} has duplicates"


def test_generator_is_reproducible(tmp_path):
    out = tmp_path / "regen.json"
    subprocess.run([sys.executable, "src/data/make_kshot_split.py", "--out", str(out)],
                   cwd=REPO, check=True, capture_output=True)
    a = json.loads(out.read_text())["splits"]
    b = json.loads((REPO / "configs" / "kshot_splits.json").read_text())["splits"]
    assert a == b, "regenerating the splits changed them"


# --- experiment design ----------------------------------------------------

def test_held_out_suite_does_not_leak_into_stage1():
    inv = json.loads((REPO / "results" / "dataset_inventory.json").read_text())
    stage1 = set(json.loads((REPO / "configs" / "stage1_split.json").read_text())
                 ["stage1_train_episodes"])
    held_out = set(inv["per_suite"]["libero_object"]["episode_indices"])
    assert not (stage1 & held_out), "libero_object episodes leaked into stage-1 training"


def test_operator_time_uses_the_control_rate_not_the_fps_field(splits):
    """The dataset says fps=10; the real control rate is 20 Hz (docs/02 §3)."""
    inv = json.loads((REPO / "results" / "dataset_inventory.json").read_text())
    assert inv["fps_metadata"] == 10.0, "dataset metadata changed"
    assert splits["control_hz"] == 20.0, "operator time must use the control rate"
    mean_frames = sum(t["frames_mean"] for t in inv["libero_object_per_task"].values()) / 10
    assert abs(splits["suite_mean_demo_seconds"] - mean_frames / 20.0) < 0.01


def test_operator_minutes_are_linear_in_k(splits):
    per_demo = splits["suite_mean_demo_seconds"] + splits["reset_seconds_assumed"]
    for name, spec in splits["splits"].items():
        expected = round(spec["K"] * per_demo / 60, 2)
        assert abs(spec["operator_time"]["minutes_per_task"] - expected) < 0.01, name


# --- eval harness ---------------------------------------------------------

def test_recorded_seeds_match_the_derivation():
    """seed = start_seed + episode_ix, the formula run_eval.py reconstructs."""
    runs = sorted((REPO / "results").glob("stage2_k*_seed*/results.json"))
    assert runs, "no eval results to check"
    for f in runs:
        d = json.loads(f.read_text())
        start = d["eval"]["start_seed"]
        for e in d["episodes"]:
            assert e["seed"] == start + e["episode_ix"], f"{f.parent.name} {e}"


def test_every_run_used_the_frozen_configuration():
    runs = sorted((REPO / "results").glob("stage2_k*_seed*/results.json"))
    for f in runs:
        d = json.loads(f.read_text())
        assert d["policy"]["n_action_steps"] == 10
        assert d["env"]["control_mode"] == "relative"
        assert d["env"]["init_states"] is True
        assert d["env"]["hard_reset"] is True
        assert d["eval"]["batch_size"] == 1, "batch_size 0 would be CPU-count dependent"


# --- analysis guards ------------------------------------------------------

def test_analysis_refuses_to_pool_mixed_resolutions():
    """results/ holds both the 5- and 10-episode sweeps; the default glob matches
    both, and averaging them would produce a plausible curve of nothing."""
    r = subprocess.run(
        [sys.executable, "src/analysis/plot_curve.py",
         "--pattern", "results/stage2_k*_seed*"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0, "mixed-resolution pooling was not refused"
    assert "different resolutions" in (r.stdout + r.stderr)
