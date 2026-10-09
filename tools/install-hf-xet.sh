#!/bin/bash
# install-hf-xet.sh - set up hf_xet (HuggingFace XET fast-download backend).
#
# Why this exists: system Python on CachyOS/Arch is PEP 668 externally-managed,
# so `pip install hf_xet` globally fails. This installer puts everything in a
# private venv at ~/strata/venv and never touches system Python (no sudo).
#
# Usage: bash tools/install-hf-xet.sh [--recreate]
#   --recreate  wipe and rebuild the venv from scratch.
#
# After install:
#   ~/strata/venv/bin/hf download <repo> <file>   # fast XET-backed download
#   ~/strata/venv/bin/python -c "import hf_xet"    # backend import check
set -euo pipefail

VENV="$HOME/strata/venv"
PY="python3"

recreate=0
for arg in "$@"; do
  case "$arg" in
    --recreate) recreate=1 ;;
    -h|--help)
      sed -n '2,/^set /p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[X] $PY not found; install python first." >&2
  exit 1
fi
"$PY" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "[X] Python 3.10+ with venv required (have: $("$PY" --version 2>&1))." >&2; exit 1; }

if [ "$recreate" -eq 1 ] && [ -d "$VENV" ]; then
  echo "[*] --recreate: removing $VENV"
  rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[*] creating venv at $VENV (using bundled ensurepip, no sudo needed)"
  "$PY" -m venv "$VENV" || {
    echo "[X] venv creation failed. On Arch/CachyOS try: sudo pacman -S python-virtualenv" >&2
    exit 1
  }
else
  echo "[*] reusing existing venv at $VENV"
fi

echo "[*] upgrading pip"
"$VENV/bin/pip" install -q -U pip

echo "[*] installing huggingface_hub[cli] + hf_xet"
"$VENV/bin/pip" install -U "huggingface_hub[cli]" hf_xet

echo "[*] verifying"
"$VENV/bin/python" -c "import huggingface_hub, hf_xet; print('huggingface_hub', huggingface_hub.__version__)"
"$VENV/bin/hf" --help >/dev/null && echo "[ok] hf CLI works: $VENV/bin/hf"

echo "[*] linking hf onto PATH (~/.local/bin, no sudo needed)"
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/hf" "$HOME/.local/bin/hf"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) echo "[ok] ~/.local/bin is on PATH" ;;
  *) echo "[!] ~/.local/bin is NOT on PATH; add: export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

cat <<EOF

[ok] hf_xet ready.
  hf CLI      : hf download <repo-id> <filename>   (via ~/.local/bin/hf)
  venv python : $VENV/bin/python
  activate    : source $VENV/bin/activate
  reinstall   : bash tools/install-hf-xet.sh --recreate
EOF
