#!/usr/bin/env bash
# Set up the official Atlas-of-ICL code (Kahardipraja et al., 2025) as its README asks:
# clone at a pinned commit, `uv sync` its locked env, and put NQ-Swap where its loader looks.
# Also installs what the env needs on this machine but does not ship: a JDK 11 (pyserini is
# imported unconditionally by model_src) and spaCy's en_core_web_trf (used by score_bem.py).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ATLAS=$ROOT/third_party/in-context-atlas
COMMIT=a30e468140
JDK=$HOME/.local/jdk-11
PY=$ATLAS/.venv/bin/python

[ -d "$ATLAS" ] || git clone https://github.com/pkhdipraja/in-context-atlas "$ATLAS"
git -C "$ATLAS" checkout -q $COMMIT
(cd "$ATLAS" && uv sync)

if [ ! -x "$JDK/bin/java" ]; then
  tmp=$(mktemp -d)
  curl -sSL -o "$tmp/jdk.tar.gz" \
    https://api.adoptium.net/v3/binary/latest/11/ga/linux/x64/jdk/hotspot/normal/eclipse
  mkdir "$tmp/x" && tar xzf "$tmp/jdk.tar.gz" -C "$tmp/x"
  mv "$tmp"/x/jdk-11* "$JDK" && rm -rf "$tmp"
fi

# Not in the lock file; installed into the venv directly. Run the scripts with $PY, not
# `uv run`, so a re-sync does not remove it.
uv pip install --python "$PY" \
  "en_core_web_trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"

# NQ-Swap (pminervini/NQ-Swap, split "dev"), saved with the env's own `datasets` so the
# official `load_from_disk` reads it. `smoke` is the first 16 rows, for a quick end-to-end run.
"$PY" - "$ROOT/data" <<'EOF'
import sys
from pathlib import Path
from datasets import DatasetDict, load_dataset

data = Path(sys.argv[1])
ds = load_dataset("pminervini/NQ-Swap")
ds.save_to_disk(data / "full" / "qa_dataset" / "NQ-Swap")
DatasetDict({"dev": ds["dev"].select(range(16))}).save_to_disk(data / "smoke" / "qa_dataset" / "NQ-Swap")
print({k: len(v) for k, v in ds.items()})
EOF
