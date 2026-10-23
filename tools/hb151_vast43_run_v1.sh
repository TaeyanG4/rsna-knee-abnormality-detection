#!/usr/bin/env bash
# HB151 Vast seed43 launcher v1.0.0; no polling, no implicit resume.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode=${1:-}
case "$mode" in smoke|fit) ;; *) echo 'Usage: bash run.sh smoke | fit [--resume]' >&2; exit 2;; esac
if [[ $# -gt 2 || ( $# -eq 2 && ( "$mode" != fit || "$2" != --resume ) ) ]]; then exit 2; fi
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -B -m tools.hb151_vast43_v1 verify
log="stage-seed43-v1/launch-${mode}-$(date -u +%Y%m%dT%H%M%S)-$$.log"
(set -o noclobber; : > "$log")
nohup python -u -B -m tools.hb151_vast43_v1 "$@" >> "$log" 2>&1 < /dev/null &
printf 'STARTED supervisor_pid=%s log=%s (admission outcome is in the log)\n' "$!" "$log"
printf 'Heartbeat: stage-seed43-v1/%s/heartbeat.json; final receipt: same directory/receipt.json\n' \
  "$([[ "$mode" == smoke ]] && printf smoke-attempt || printf 'fit-attempt-N')"
