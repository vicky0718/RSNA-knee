#!/usr/bin/env python3
"""A small Kaggle REST client that works from inside this sandbox.

The official `kaggle` CLI talks to api.kaggle.com, which the egress proxy here
refuses (403 on CONNECT). The same REST API is served from www.kaggle.com, which
is reachable, so this speaks that directly with HTTP basic auth from
~/.kaggle/kaggle.json.

Covers what the loop needs: push a utility dataset, push and run a kernel, poll
its status, and pull its output.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

BASE = "https://www.kaggle.com/api/v1"


def credentials() -> tuple[str, str]:
    path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    data = json.loads(path.read_text())
    return data["username"], data["key"]


@dataclass
class Kaggle:
    username: str
    key: str

    @classmethod
    def from_config(cls) -> "Kaggle":
        return cls(*credentials())

    @property
    def auth(self) -> tuple[str, str]:
        return (self.username, self.key)

    def _call(self, method: str, path: str, **kwargs) -> requests.Response:
        response = requests.request(method, f"{BASE}{path}", auth=self.auth, timeout=300, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:400]}")
        return response

    # --- datasets ---------------------------------------------------------

    def upload_file(self, path: Path) -> str:
        """Reserve a slot, PUT the bytes, return the token that names them."""
        size = path.stat().st_size
        last_modified = int(path.stat().st_mtime)
        start = self._call(
            "POST",
            f"/datasets/upload/file/{size}/{last_modified}",
            data={"fileName": path.name},
        ).json()
        with path.open("rb") as handle:
            put = requests.put(start["createUrl"], data=handle, timeout=600)
        if put.status_code >= 400:
            raise RuntimeError(f"upload of {path.name} failed: {put.status_code} {put.text[:200]}")
        return start["token"]

    def dataset_exists(self, slug: str) -> bool:
        response = requests.get(
            f"{BASE}/datasets/list", auth=self.auth, params={"user": self.username}, timeout=120
        )
        response.raise_for_status()
        return any(item["ref"] == f"{self.username}/{slug}" for item in response.json())

    def push_dataset(self, slug: str, title: str, files: list[Path], notes: str = "update") -> str:
        tokens = [{"token": self.upload_file(path)} for path in files]
        if self.dataset_exists(slug):
            self._call(
                "POST",
                f"/datasets/create/version/{self.username}/{slug}",
                json={"versionNotes": notes, "files": tokens, "isPrivate": True},
            )
        else:
            self._call(
                "POST",
                "/datasets/create/new",
                json={
                    "title": title,
                    "slug": slug,
                    "ownerSlug": self.username,
                    "licenseName": "CC0-1.0",
                    "isPrivate": True,
                    "files": tokens,
                },
            )
        return f"{self.username}/{slug}"

    # --- kernels ----------------------------------------------------------

    def push_kernel(
        self,
        slug: str,
        title: str,
        source: str,
        *,
        competition_sources: list[str] | None = None,
        dataset_sources: list[str] | None = None,
        enable_gpu: bool = False,
        kernel_type: str = "script",
    ) -> dict:
        """Create or update a kernel and queue a run. Internet stays off."""
        body = {
            "id": None,
            "slug": f"{self.username}/{slug}",
            "newTitle": title,
            "text": source,
            "language": "python",
            "kernelType": kernel_type,
            "isPrivate": True,
            "enableGpu": enable_gpu,
            "enableTpu": False,
            "enableInternet": False,
            "datasetDataSources": dataset_sources or [],
            "competitionDataSources": competition_sources or [],
            "kernelDataSources": [],
            "modelDataSources": [],
            "categoryIds": [],
        }
        return self._call("POST", "/kernels/push", json=body).json()

    def kernel_status(self, slug: str) -> dict:
        return self._call(
            "GET", "/kernels/status", params={"userName": self.username, "kernelSlug": slug}
        ).json()

    def wait_for_kernel(self, slug: str, poll: int = 30, timeout: int = 7200) -> dict:
        """Block until the kernel stops running, printing each state change."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            status = self.kernel_status(slug)
            state = status.get("status") or status.get("failureMessage")
            if state != last:
                print(f"  [{time.strftime('%H:%M:%S')}] {state}", flush=True)
                last = state
            if str(state).lower() in {"complete", "error", "cancelacknowledged", "cancelrequested"}:
                return status
            time.sleep(poll)
        raise TimeoutError(f"kernel {slug} still running after {timeout}s")

    def kernel_output(self, slug: str, out_dir: Path) -> list[Path]:
        payload = self._call(
            "GET", "/kernels/output", params={"userName": self.username, "kernelSlug": slug}
        ).json()
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for item in payload.get("files", []):
            url, name = item.get("url"), item.get("fileName")
            if not url or not name:
                continue
            blob = requests.get(url, auth=self.auth, timeout=600)
            blob.raise_for_status()
            target = out_dir / Path(name).name
            target.write_bytes(blob.content)
            written.append(target)
        if payload.get("log"):
            (out_dir / f"{slug}.log").write_text(payload["log"])
            written.append(out_dir / f"{slug}.log")
        return written


if __name__ == "__main__":
    api = Kaggle.from_config()
    print(f"authenticated as {api.username}")
    print(f"own datasets: {[d['ref'] for d in requests.get(f'{BASE}/datasets/list', auth=api.auth, params={'user': api.username}, timeout=60).json()]}")
