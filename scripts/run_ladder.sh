#!/usr/bin/env bash
# Benchmark ladder stages one after another, re-running any stage whose result is flagged
# (clock drop, clock below rated boost, thermal slowdown).
#   scripts/run_ladder.sh                          -> all configs/stages/*.yaml
#   scripts/run_ladder.sh s2_mem_efficient s3_compile
# On the laptop the GPU can be slowed by the platform's thermal management (SW thermal
# slowdown, memory clock 6801 -> 810 MHz) while the GPU itself reads under 50 C, so a
# flagged stage gets a longer rest before the next attempt, not just a GPU-temperature check.
set -uo pipefail
cd "$(dirname "$0")/.."

ATTEMPTS=${ATTEMPTS:-3}
REST_S=${REST_S:-300}
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=($(ls configs/stages/*.yaml | xargs -n1 basename | sed 's/\.yaml$//'))

gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
cool_down() { while [ "$(gpu_temp)" -ge 50 ]; do sleep 10; done; }

mkdir -p runs
for stage in "${STAGES[@]}"; do
  for attempt in $(seq 1 "$ATTEMPTS"); do
    cool_down
    echo "=== $stage attempt $attempt, $(date +%T), GPU $(gpu_temp) C"
    log=runs/bench_${stage}.log
    scripts/run_container.sh python scripts/bench_train.py --config "configs/stages/$stage.yaml" > "$log" 2>&1
    grep -E "^(tokens/s|MFU|peak memory|WARNING|saved)|Traceback|Error" "$log" | grep -v warnings.warn
    out=$(grep -oP '^saved\s+\K\S+' "$log") || { echo "=== $stage failed, see $log"; break; }
    flags=$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['throttle_flags']))" "$out")
    [ "$flags" -eq 0 ] && { echo "=== $stage clean"; break; }
    if [ "$attempt" -lt "$ATTEMPTS" ]; then
      echo "=== $stage flagged; resting ${REST_S} s"; sleep "$REST_S"
    else
      echo "=== $stage still flagged after $ATTEMPTS attempts; kept the last result"
    fi
  done
done
