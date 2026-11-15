"""Build the Colab bundle: one zip with everything the notebook needs.

    uv run python data/build_colab_bundle.py [--out ../colab_bundle.zip]

Contents: the files git tracks (so nothing local or private can slip in, and never `.env`)
plus three assets git does not hold - the extended catalog, the E3 model and the caches the
notebook replays from (Claude, TMDB, document vectors). Top-level folder in the zip:
`emberfall_detective/`, which the notebook's setup cell expects. The zip is written outside
the repository; upload it to Google Drive and share it by link (docs/COLAB_ASSETS.md).
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

from concierge.config import REPO_ROOT

ASSETS = ["data/catalog_extended.parquet", "models/E3", ".cache/llm", ".cache/tmdb",
          ".cache/vectors", ".cache/llm_usage.jsonl"]  # fmt: skip
TOP = "emberfall_detective"


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True)
    return [Path(line) for line in out.stdout.decode().splitlines()]


def assets() -> list[Path]:
    found = []
    for asset in ASSETS:
        path = REPO_ROOT / asset
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        found += [p.relative_to(REPO_ROOT) for p in files if p.name != ".DS_Store"]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "colab_bundle.zip")
    args = parser.parse_args()
    paths = tracked() + assets()
    assert not any(p.name == ".env" for p in paths)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in paths:
            z.write(REPO_ROOT / rel, f"{TOP}/{rel.as_posix()}")
    print(f"wrote {args.out} ({len(paths)} files, {args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
