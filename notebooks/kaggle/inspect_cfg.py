"""Dump the full config of the fold models, and the per-target scores of the rest.

Two questions the first pass left open:

1. The A5 fold checkpoints carry a `cfg` with `meta`, `n_meta` and `n_sites`
   keys. If those models take *site identity as an input feature*, they are not
   merely vulnerable to the scanner leak measured in docs/results.md §4 — they
   are built on it, and whatever the private split does with sites decides what
   happens to them.
2. The CoAtNet bundles report `gold_auc` and a per-target `aucs` dict but ship
   no OOF. Reading those tells us the relative strength of the arms we cannot
   refit, which is what decides whether refitting the ones we can is worth it.
"""

import json
from pathlib import Path

import torch

ROOT = Path("/kaggle/input")


def show(path, keys):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {path.name}")
    for key in keys:
        if key not in blob:
            continue
        value = blob[key]
        if isinstance(value, dict) and not any(torch.is_tensor(v) for v in value.values()):
            print(f"  {key}: {json.dumps(value, default=str)[:1200]}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with tensors, keys={list(value)[:20]}")
        else:
            print(f"  {key}: {str(value)[:600]}")
    del blob


for path in sorted(ROOT.rglob("m_f*.pt"))[:2]:
    show(path, ["cfg", "fold"])

for path in sorted(ROOT.rglob("raptor_ft_coatnet_*.pt")):
    show(path, ["gold_auc", "aucs", "arch", "res"])

for path in sorted(ROOT.rglob("manifest.json"))[:3]:
    payload = json.loads(path.read_text())
    print(f"\n=== {path.parent.name}/{path.name}")
    for key, value in payload.items():
        if key == "members" and isinstance(value, list):
            print(f"  members: {len(value)}")
            for member in value[:4]:
                print(f"    {json.dumps(member, default=str)[:300]}")
        else:
            print(f"  {key}: {json.dumps(value, default=str)[:400]}")

print("\ndone")
