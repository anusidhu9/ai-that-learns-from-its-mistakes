"""Build the Hugging Face Space from this repository and push it.

    python space/deploy.py --space anusidhu9/mind --stage-only   # build, don't push
    python space/deploy.py --space anusidhu9/mind                # build and push

The Space holds only what the app needs: src/, results/, LICENSE, and the
Space's own README.md (with its settings header), app.py and requirements.txt
from this folder. The Space must already exist (create it on huggingface.co
with the Gradio SDK). Pushing uses git, so the first push asks for your
Hugging Face username and an access token with write permission.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_FILES = Path(__file__).resolve().parent


KEEP = {".git", ".gitattributes"}  # Hugging Face's git metadata and LFS rules


def build(stage: Path) -> None:
    """Make `stage` contain exactly the app's files.

    Everything else is removed, including files from a Space template
    (such as the static template's index.html and style.css).
    """
    for target in stage.iterdir():
        if target.name in KEEP:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    # Python files only: anything else in src/ (caches, or a stray file
    # holding a secret such as a token) must never be published.
    shutil.copytree(ROOT / "src", stage / "src",
                    ignore=lambda _dir, names: [n for n in names if not n.endswith(".py")])
    shutil.copytree(ROOT / "results", stage / "results")
    shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
    for name in ("README.md", "app.py", "requirements.txt"):
        shutil.copy2(SPACE_FILES / name, stage / name)


def git(*args: str, cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and push the MIND Hugging Face Space.")
    parser.add_argument("--space", required=True, help="Space id, e.g. anusidhu9/mind")
    parser.add_argument("--stage-only", action="store_true",
                        help="build into a local folder without pushing")
    parser.add_argument("--stage", help="folder to build into (default: a new temp folder)")
    args = parser.parse_args()

    url = f"https://huggingface.co/spaces/{args.space}"
    stage = Path(args.stage) if args.stage else Path(tempfile.mkdtemp(prefix="mind_space_"))
    if args.stage_only:
        stage.mkdir(parents=True, exist_ok=True)
        build(stage)
        print(f"Built the Space in {stage}")
        return

    # Clone the existing Space so its history (and Hugging Face's own
    # .gitattributes) is kept, then replace the app files and push.
    if not (stage / ".git").exists():
        if stage.exists() and any(stage.iterdir()):
            raise SystemExit(f"{stage} is not empty and not a git clone")
        subprocess.run(["git", "clone", url, str(stage)], check=True)
    else:
        git("pull", "--ff-only", cwd=stage)
    build(stage)
    git("add", "-A", cwd=stage)
    if git("status", "--porcelain", cwd=stage):
        revision = git("rev-parse", "--short", "HEAD", cwd=ROOT)
        git("commit", "-m", f"Deploy MIND dashboard from GitHub {revision}", cwd=stage)
    # Push whenever the Space lacks local commits, including a commit left
    # behind by an earlier attempt whose push failed.
    if git("rev-list", "--count", "@{upstream}..HEAD", cwd=stage) == "0":
        print("The Space is already up to date.")
        return
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=stage, check=True)
    print(f"Pushed. The Space rebuilds automatically: {url}")


if __name__ == "__main__":
    main()
