"""Look inside the public weight bundles for a recoverable fold assignment.

The public arms were trained on all 4,407 studies, so their predictions on
training data are partly memorised and cannot be used to fit fusion weights
honestly. That changes if a bundle records *which studies each fold held out* —
then genuine out-of-fold predictions can be assembled from the checkpoints we
already have, and the per-finding routing can be refitted against something real
instead of against a 30% leaderboard split.

So this reads metadata only: every non-tensor value in every checkpoint, plus any
JSON or CSV shipped alongside. It is looking for one of four things, in
descending order of usefulness:

  1. saved OOF predictions (nothing left to compute)
  2. an explicit study -> fold mapping
  3. a fold seed plus the splitter used, enough to reproduce the split
  4. nothing, in which case honest fusion needs our own arm

Metadata only: no pixels, no weights printed, nothing copied out.
"""

import json
import os
from pathlib import Path

import torch

ROOT = Path("/kaggle/input")
INTERESTING = ("fold", "oof", "valid", "split", "study", "uid", "index", "seed", "cv", "holdout")
MAX_PT_BYTES = 3 * 1024**3


def describe(value, depth=0):
    """A short, safe description of a checkpoint value."""
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, dict):
        keys = list(value)[:14]
        return f"dict({len(value)}) keys={keys}"
    if isinstance(value, (list, tuple)):
        head = value[:4]
        kinds = {type(v).__name__ for v in value[:50]}
        return f"{type(value).__name__}({len(value)}) of {kinds} head={str(head)[:120]}"
    text = str(value)
    return text[:160] if len(text) <= 160 else text[:160] + "..."


def scan_checkpoint(path):
    size = path.stat().st_size
    if size > MAX_PT_BYTES:
        print(f"  [skip, {size / 1e9:.1f} GB] {path.name}")
        return
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"  [unreadable] {path.name}: {type(error).__name__}: {error}")
        return
    if not isinstance(blob, dict):
        print(f"  {path.name}: top-level {type(blob).__name__}")
        return
    non_tensor = {k: v for k, v in blob.items() if not torch.is_tensor(v) and k != "model"}
    print(f"  {path.name} ({size / 1e6:.0f} MB) keys={list(blob)[:14]}")
    for key, value in list(non_tensor.items())[:14]:
        marker = " <<<" if any(word in str(key).lower() for word in INTERESTING) else ""
        print(f"      {key}: {describe(value)}{marker}")
    del blob


def scan_sidecar(path):
    name = path.name.lower()
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except Exception as error:
            print(f"  [bad json] {path.name}: {error}")
            return
        keys = list(payload) if isinstance(payload, dict) else f"list({len(payload)})"
        marker = " <<<" if any(w in name for w in INTERESTING) else ""
        print(f"  {path.name}: {keys}{marker}")
        if isinstance(payload, dict):
            for key, value in list(payload.items())[:10]:
                if any(w in str(key).lower() for w in INTERESTING):
                    print(f"      {key}: {describe(value)} <<<")
    elif path.suffix == ".csv":
        with path.open() as handle:
            header = handle.readline().strip()
            rows = sum(1 for _ in handle)
        marker = " <<<" if any(w in header.lower() or w in name for w in INTERESTING) else ""
        print(f"  {path.name}: {rows} rows | {header[:150]}{marker}")


for dataset in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if "competition" in dataset.name.lower():
        continue
    files = sorted(Path(dataset).rglob("*"))
    checkpoints = [f for f in files if f.suffix in (".pt", ".pth", ".ckpt")]
    sidecars = [f for f in files if f.suffix in (".json", ".csv")]
    if not checkpoints and not sidecars:
        continue
    print(f"\n=== {dataset.name}: {len(checkpoints)} checkpoints, {len(sidecars)} sidecars")
    for path in sidecars[:25]:
        scan_sidecar(path)
    for path in checkpoints[:8]:
        scan_checkpoint(path)

print("\ndone")
