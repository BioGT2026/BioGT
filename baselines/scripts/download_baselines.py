#!/usr/bin/env python
"""Clone pinned upstream baseline sources and apply BioGT compatibility patches."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
PATCHES = ROOT / "patches"
SOURCES = {
    "KnowledgeGraphEmbedding": {
        "url": "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding.git",
        "commit": "2e440e0f9c687314d5ff67ead68ce985dc446e3a",
        "patch": "knowledge_graph_embedding_joint4.patch",
        "added_files": (),
    },
    "CompGCN": {
        "url": "https://github.com/malllabiisc/CompGCN.git",
        "commit": "3b06f5fec42526faa81afc158df9e64a8382982c",
        "patch": "compgcn_joint4.patch",
        "added_files": ("torch_scatter.py",),
    },
    "NBFNet-PyG": {
        "url": "https://github.com/KiddoZhu/NBFNet-PyG.git",
        "commit": "0c2d262e1d0175c174f62367d08ce4060b0d7b89",
        "patch": "nbfnet_pyg_compat.patch",
        "added_files": ("nbfnet/scatter_compat.py",),
    },
}


def command(*args: str, cwd: Path | None = None, check: bool = True):
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def ensure_source(name: str, spec: dict[str, str], reset: bool) -> None:
    destination = EXTERNAL / name
    if not destination.exists():
        command("git", "clone", spec["url"], str(destination))
        command("git", "checkout", "--detach", spec["commit"], cwd=destination)
    elif not (destination / ".git").exists():
        if not reset:
            raise SystemExit(f"{destination} exists but is not a Git checkout")
        backup = destination.with_name(f"{destination.name}.legacy_pre_pinned")
        if backup.exists():
            raise SystemExit(
                f"Cannot preserve non-Git source because backup exists: {backup}"
            )
        destination.rename(backup)
        print(f"{name}: preserved legacy source at {backup}")
        command("git", "clone", spec["url"], str(destination))
        command("git", "checkout", "--detach", spec["commit"], cwd=destination)

    if reset:
        for relative_path in spec["added_files"]:
            generated_path = destination / relative_path
            tracked = command(
                "git",
                "ls-files",
                "--error-unmatch",
                relative_path,
                cwd=destination,
                check=False,
            ).returncode == 0
            if generated_path.exists() and not tracked:
                generated_path.unlink()
        command("git", "reset", "--hard", spec["commit"], cwd=destination)

    head = command("git", "rev-parse", "HEAD", cwd=destination).stdout.strip()
    if head != spec["commit"]:
        raise SystemExit(
            f"{name} is at {head}; expected pinned upstream {spec['commit']}"
        )

    patch_path = PATCHES / spec["patch"]
    check_result = command(
        "git",
        "apply",
        "--check",
        "--unidiff-zero",
        "--ignore-space-change",
        str(patch_path),
        cwd=destination,
        check=False,
    )
    if check_result.returncode == 0:
        command(
            "git",
            "apply",
            "--unidiff-zero",
            "--ignore-space-change",
            str(patch_path),
            cwd=destination,
        )
        print(f"{name}: applied {patch_path.name}")
        return

    reverse_check = command(
        "git",
        "apply",
        "--reverse",
        "--check",
        "--unidiff-zero",
        "--ignore-space-change",
        str(patch_path),
        cwd=destination,
        check=False,
    )
    if reverse_check.returncode == 0:
        print(f"{name}: patch already applied")
        return
    raise SystemExit(
        f"{name}: checkout is dirty or incompatible with {patch_path.name}\n"
        f"{check_result.stderr}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset only tracked vendor files to the pinned commit before patching",
    )
    args = parser.parse_args()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        ensure_source(name, spec, args.reset)


if __name__ == "__main__":
    main()
