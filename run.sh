#!/bin/bash
# Fish Audio S2 Pro Wyoming bridge launcher.
#
# First run: installs uv if missing, clones fish-speech at a pinned commit,
# builds the venv and downloads the model (~11 GB). Later runs skip straight
# to starting the bridge.
#
# Configuration: optional .env next to this script (see .env.example).
# Extra arguments are passed through to bridge_s2.py, e.g.
#   ./run.sh --temperature 0.8
#
#   SETUP_ONLY=1 ./run.sh   set everything up, don't start the bridge
#   REINSTALL=1 ./run.sh    reinstall Python dependencies
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f .env ]; then
    set -a
    source <(tr -d '\r' < .env)
    set +a
fi

FISH_SPEECH_COMMIT="541566932abbaebcb0a40e562a1f846738383f38"
FISH_SPEECH_DIR="src/fish-speech"
TORCH_EXTRA="${TORCH_EXTRA:-cu128}"
MODEL_ID="${MODEL_ID:-fishaudio/s2-pro}"
MODEL_DIR="${MODEL_DIR:-models/s2-pro}"
PROMPT_WAV="${PROMPT_WAV-/root/cosyvoice-wyoming/voices/nicoleheart.wav}"
PROMPT_TEXT="${PROMPT_TEXT-The day had gently ended. In the soft room, the person lay down. Slowly, the light faded over the floor. Eyes drifted shut and the body relaxed. Thoughts slowed and turned soft. Warmth spread through the sheets and the body let go, drifting into sleep.}"
VOICE="${VOICE:-nicoleheart}"
URI="${URI:-tcp://0.0.0.0:10215}"
INT8="${INT8:-1}"
COMPILE="${COMPILE:-1}"
DEFAULT_MOOD="${DEFAULT_MOOD-}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.7}"
SEED="${SEED:-42}"
PREBUFFER_SECONDS="${PREBUFFER_SECONDS:-1}"
MIN_MERGED_SECONDS="${MIN_MERGED_SECONDS:-3}"

for var in TEMPERATURE TOP_P SEED PREBUFFER_SECONDS MIN_MERGED_SECONDS; do
    if ! [[ "${!var}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: $var='${!var}' is not a number (check .env)" >&2
        exit 1
    fi
done

# torch.compile cache: /tmp is cleared at boot, which would mean a cold
# compile (several minutes) on every start.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DIR/.cache/inductor}"

# Only needed for setup; systemd's PATH usually lacks ~/.local/bin.
ensure_uv() {
    export PATH="${HOME:-/root}/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo ">>> Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

if [ ! -d "$FISH_SPEECH_DIR/.git" ]; then
    echo ">>> Cloning fish-speech..."
    git clone https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_DIR"
fi
if [ "$(git -C "$FISH_SPEECH_DIR" rev-parse HEAD)" != "$FISH_SPEECH_COMMIT" ]; then
    echo ">>> Checking out fish-speech $FISH_SPEECH_COMMIT..."
    git -C "$FISH_SPEECH_DIR" fetch -q origin "$FISH_SPEECH_COMMIT" || true
    git -C "$FISH_SPEECH_DIR" checkout -q "$FISH_SPEECH_COMMIT"
fi

STAMP=".venv/.deps-$FISH_SPEECH_COMMIT-$TORCH_EXTRA"
if [ ! -f "$STAMP" ] || [ "${REINSTALL:-0}" = 1 ]; then
    ensure_uv
    echo ">>> Installing dependencies..."
    # pyaudio is only used by fish-speech's API client and needs portaudio headers
    if (cd "$FISH_SPEECH_DIR" && UV_PROJECT_ENVIRONMENT="$DIR/.venv" \
            uv sync --frozen --no-dev --extra "$TORCH_EXTRA" \
            --no-install-package pyaudio) \
        && uv pip install -p .venv wyoming==1.10.2 soundfile "huggingface_hub[hf_xet]"; then
        rm -f .venv/.deps-*
        touch "$STAMP"
    elif [ -x .venv/bin/python ] && .venv/bin/python -c "import torch, wyoming" 2>/dev/null; then
        echo "WARNING: dependency install failed; starting with what is installed" >&2
    else
        echo "ERROR: dependency install failed" >&2
        exit 1
    fi
fi

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] || [ ! -f "$MODEL_DIR/codec.pth" ]; then
    echo ">>> Downloading $MODEL_ID (~11 GB)..."
    .venv/bin/python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
" "$MODEL_ID" "$MODEL_DIR"
fi

if [ -n "$PROMPT_WAV" ] && [ ! -f "$PROMPT_WAV" ]; then
    echo "ERROR: reference voice $PROMPT_WAV not found." >&2
    echo "Set PROMPT_WAV in .env to a clean 10-30 s WAV of the voice to clone" >&2
    echo "(and PROMPT_TEXT to its exact transcript), or PROMPT_WAV= for no cloning." >&2
    exit 1
fi

if [ "${SETUP_ONLY:-0}" = 1 ]; then
    echo ">>> Setup complete."
    exit 0
fi

ARGS=(
    --fish-speech-repo "$FISH_SPEECH_DIR"
    --model-dir "$MODEL_DIR"
    --prompt-wav "$PROMPT_WAV"
    --prompt-text "$PROMPT_TEXT"
    --voice "$VOICE"
    --uri "$URI"
    --default-mood "$DEFAULT_MOOD"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --seed "$SEED"
    --prebuffer-seconds "$PREBUFFER_SECONDS"
    --min-merged-seconds "$MIN_MERGED_SECONDS"
)
[ "$INT8" = 1 ] && ARGS+=(--int8)
[ "$COMPILE" = 1 ] || ARGS+=(--no-compile)

# EXTRA_ARGS from .env: a plain string ("--top-k 20") or a bash array
if declare -p EXTRA_ARGS 2>/dev/null | grep -q '^declare -a'; then
    ARGS+=("${EXTRA_ARGS[@]}")
elif [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a _extra <<< "$EXTRA_ARGS"
    ARGS+=("${_extra[@]}")
fi

CMD=(.venv/bin/python bridge_s2.py "${ARGS[@]}" "$@")
echo ">>> cmd: ${CMD[*]}"
exec "${CMD[@]}"
