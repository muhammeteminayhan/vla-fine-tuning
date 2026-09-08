#!/usr/bin/env bash
# Run a command while sampling GPU memory, and report the peak.
#
# CLAUDE.md requires peak VRAM to be logged for every training run, and it is
# just as useful for eval. nvidia-smi is sampled from outside the process so
# this works for any command, not only Python.
#
# Usage: scripts/run_with_gpu_peak.sh <label> <logfile> <cmd...>
set -u

LABEL="$1"; shift
LOGFILE="$1"; shift

SAMPLES="$(mktemp)"
"$@" > "$LOGFILE" 2>&1 &
CMD_PID=$!

(
  while kill -0 "$CMD_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
    sleep 0.25
  done
) > "$SAMPLES" &
SAMPLER_PID=$!

START=$(date +%s.%N)
wait "$CMD_PID"; RC=$?
END=$(date +%s.%N)
wait "$SAMPLER_PID" 2>/dev/null

PEAK=$(sort -n "$SAMPLES" | tail -1)
BASE=$(sort -n "$SAMPLES" | head -1)
N=$(wc -l < "$SAMPLES")
rm -f "$SAMPLES"

python3 - "$LABEL" "$RC" "$START" "$END" "${PEAK:-0}" "${BASE:-0}" "$N" <<'PY'
import json, sys
label, rc, start, end, peak, base, n = sys.argv[1:8]
print(json.dumps({
    "label": label,
    "exit_code": int(rc),
    "wall_clock_s": round(float(end) - float(start), 2),
    "gpu_peak_mib": int(peak),
    "gpu_baseline_mib": int(base),
    "gpu_samples": int(n),
}))
PY
exit $RC
