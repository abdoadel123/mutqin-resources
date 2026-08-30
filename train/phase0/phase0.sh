#!/usr/bin/env bash
# Phase 0 of STREAMING_PLAN.md — one command, two modes:
#
#   phase0.sh mechanics <wav...>   FREE half: downloads NVIDIA's public cache-aware streaming
#                                  FastConformer (English) and proves the whole mechanics —
#                                  cache export, chunked/full parity, stream_meta.json — with
#                                  zero training. Any 16 kHz mono speech wav works (English
#                                  audio for the English checkpoint).
#
#   phase0.sh convert <wav...>     The $5 half: converts OUR Quran base to streaming
#                                  (finetune.py STREAMING=1, short leg on $DATA prepared by
#                                  prep.py), then the same export + parity gates on Quran wavs.
#                                  Variables pass through: DATA (required), EPOCHS, LR, CTCW,
#                                  MAX_TIME, BASE_REPO/BASE_FILE.
#
# Exits nonzero when a gate fails — legloop-compatible. Nothing is published here:
# publishing stays a separate, deliberate act after the phase-2 judge (training.md §8).
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?الاستعمال: phase0.sh mechanics|convert [wav...]}
shift
# convert with no wavs pulls the gate set itself: real PHONE-MIC audio from R2, distinct
# devices (gate_wavs.py) — the gate must be field audio, or phase 0 repeats "the studio
# lies" at its own doorstep. mechanics still takes explicit wavs (English checkpoint).
if [ $# -eq 0 ]; then
  [ "$MODE" = convert ] || { echo 'هات ملف wav واحد على الأقل (16kHz mono)'; exit 1; }
  mapfile -t GATE < <(python gate_wavs.py "${GATE_COUNT:-6}")
  set -- "${GATE[@]}"
fi

python -c 'import nemo.collections.asr' 2>/dev/null || pip install -q 'nemo_toolkit[asr]' onnxruntime soundfile

case "$MODE" in
  mechanics)
    # The public multi-latency streaming checkpoint — same architecture family, zero training.
    MODEL=$(python - <<'EOF'
from huggingface_hub import hf_hub_download, list_repo_files
repo = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
name = next(f for f in list_repo_files(repo) if f.endswith('.nemo'))
print(hf_hub_download(repo, name))
EOF
    )
    ;;
  convert)
    : "${DATA:?convert محتاج DATA (مخرج prep.py)}"
    STREAMING=1 EPOCHS="${EPOCHS:-1}" CTCW="${CTCW:-1.0}" python ../finetune.py
    MODEL=finetuned.nemo
    ;;
  *) echo "وضع مجهول: $MODE"; exit 1 ;;
esac

python export_stream.py "$MODEL"
python stream_parity.py "$MODEL" "$@"
