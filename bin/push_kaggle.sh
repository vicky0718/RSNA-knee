#!/usr/bin/env bash
# Ship src/ to Kaggle as a private utility dataset.
#
# Notebooks attach this dataset and `sys.path.insert(0, "/kaggle/input/knee-src")`
# instead of holding pasted code: a notebook copy of a module is a fork that
# drifts, and drift between the training and inference copies is how a pipeline
# scores differently on rerun than it did in the commit.
#
# Usage: KAGGLE_USERNAME=you bin/push_kaggle.sh ["version notes"]
set -euo pipefail

notes="${1:-src update}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME (and have ~/.kaggle/kaggle.json in place)}"

cp -r "$root/src/knee" "$staging/knee"
find "$staging" -name '__pycache__' -type d -prune -exec rm -rf {} +

sed "s#USERNAME#${KAGGLE_USERNAME}#" \
    "$root/notebooks/kaggle/dataset-metadata.json" > "$staging/dataset-metadata.json"

if kaggle datasets status "${KAGGLE_USERNAME}/knee-src" >/dev/null 2>&1; then
    kaggle datasets version -p "$staging" -m "$notes" --dir-mode zip
else
    kaggle datasets create -p "$staging" --dir-mode zip
fi

echo "pushed ${KAGGLE_USERNAME}/knee-src — attach it, then:"
echo "    import sys; sys.path.insert(0, '/kaggle/input/knee-src')"
