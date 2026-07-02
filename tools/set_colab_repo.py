"""Point every 'Open in Colab' badge at YOUR GitHub repo, then rebuild notebooks.

Usage:
    python tools/set_colab_repo.py <owner>/<repo> [branch] [notebooks-subdir]

The [notebooks-subdir] is the path to the notebooks folder AS IT EXISTS IN THE REPO —
this only differs from "fable-pytorch" if you pushed the course under a different
folder name (or to the repo root, in which case pass "").

Examples:
    python tools/set_colab_repo.py hoax12/fable-pytorch
    python tools/set_colab_repo.py hoax12/fable-pytorch main fable-pytorch
    python tools/set_colab_repo.py hoax12/tensor-forge main ""   # notebooks/ at repo root
"""
import os
import sys
import subprocess

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

repo = sys.argv[1]
branch = sys.argv[2] if len(sys.argv) > 2 else "main"
subdir = sys.argv[3] if len(sys.argv) > 3 else "fable-pytorch"

env = os.environ.copy()
env["FORGE_GITHUB_REPO"] = repo
env["FORGE_GITHUB_BRANCH"] = branch
env["FORGE_GITHUB_SUBDIR"] = subdir

here = os.path.dirname(os.path.abspath(__file__))
path_preview = f"{subdir}/notebooks/..." if subdir else "notebooks/..."
print(f"Rebuilding notebooks -> github.com/{repo}/blob/{branch}/{path_preview}")
subprocess.run([sys.executable, os.path.join(here, "build_all.py")], env=env, check=True)
print("Done. Commit & push, then the Colab badges will work.")
