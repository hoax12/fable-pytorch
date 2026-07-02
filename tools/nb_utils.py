"""Notebook-building helpers for The Tensor Forge.

All course notebooks are generated programmatically (guaranteed-valid .ipynb, shared
boilerplate, one-command rebuild): `python tools/build_all.py`.
"""
from __future__ import annotations

import os
import nbformat as nbf

GITHUB_REPO = os.environ.get("FORGE_GITHUB_REPO", "your-username/tensor-forge")
GITHUB_BRANCH = os.environ.get("FORGE_GITHUB_BRANCH", "main")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(REPO_ROOT, "notebooks")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip("\n"))


def header(filename: str, act: str, number: str, title: str, tagline: str,
           prev: str | None = None, nxt: str | None = None) -> nbf.NotebookNode:
    url = (f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/"
           f"{GITHUB_BRANCH}/fable_folder/notebooks/{filename}")
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"
    nav = []
    if prev:
        nav.append(f"⬅️ [{prev.split('.')[0]}]({prev})")
    if nxt:
        nav.append(f"[{nxt.split('.')[0]}]({nxt}) ➡️")
    text = (f"{badge}\n\n# ⚒️ {act} · Quest {number} — {title}\n\n*{tagline}*\n\n")
    if nav:
        text += "  •  ".join(nav) + "\n\n"
    text += "---"
    return md(text)


# ---------------------------------------------------------------------------
# The Forge Grader — embedded in every notebook, works locally AND on Colab.
# Each notebook passes its own exercise registry (name -> (points, test_src, hint)).
# ---------------------------------------------------------------------------
GRADER_CORE = '''
# ══════════════════ ⚒️ FORGE GRADER — run me once ══════════════════
# Powers check() and xp_report(). Exercises give instant feedback + XP.
_XP = {"earned": 0, "done": set(), "checks": {}}

def _register(name, points, test, hint):
    _XP["checks"][name] = (points, test, hint)

def check(name, *answer):
    """Grade an exercise: check("ex1", your_answer). Repeatable until correct."""
    if name not in _XP["checks"]:
        print(f"unknown exercise: {name!r} — available: {list(_XP['checks'])}")
        return
    points, test, hint = _XP["checks"][name]
    try:
        ok = bool(test(*answer))
        err = None
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    if ok:
        first = name not in _XP["done"]
        if first:
            _XP["done"].add(name)
            _XP["earned"] += points
        total = sum(p for p, _, _ in _XP["checks"].values())
        tag = f"+{points} XP" if first else "already earned"
        print(f"✅ {name} — correct! {tag}   ⚡ {_XP['earned']}/{total} XP")
    else:
        msg = f"❌ {name} — not yet."
        if err:
            msg += f" (your answer raised {err})"
        print(msg + f"\\n   💡 hint: {hint}")

def xp_report():
    total = sum(p for p, _, _ in _XP["checks"].values())
    n = len(_XP["checks"])
    bar = "█" * int(10 * _XP["earned"] / max(total, 1)) or "░"
    print(f"⚡ XP: {_XP['earned']}/{total}   [{bar:<10}]   exercises: {len(_XP['done'])}/{n}")
    for name in _XP["checks"]:
        print(("  ✅ " if name in _XP["done"] else "  ⬜ ") + name)
'''


def setup_cell(extra: str = "", torch_needed: bool = True) -> nbf.NotebookNode:
    """Standard setup: imports + seeds + device + the grader core."""
    imports = ""
    if torch_needed:
        imports = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
        )
    src = (
        f"{imports}"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import math, random\n\n"
    )
    if torch_needed:
        src += (
            "torch.manual_seed(0)\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "print(f'PyTorch {torch.__version__} | device: {device}')\n"
        )
    src += "np.random.seed(0); random.seed(0)\n"
    src += "\n" + GRADER_CORE.strip("\n")
    if extra:
        src += "\n\n" + extra.strip("\n")
    return code(src)


def registry_cell(registrations: str) -> nbf.NotebookNode:
    """A cell that registers this notebook's exercises with the grader."""
    return code(
        "# ── this notebook's exercises (run once) "
        "───────────────────────────────\n" + registrations.strip("\n")
    )


def boss_md(challenges: list[str]) -> nbf.NotebookNode:
    body = (
        "## 🏆 Boss Challenges\n\n"
        "Earn your XP — write each answer in a cell below and grade it with `check(...)`. "
        "When you're done, run `xp_report()`.\n\n"
    )
    for c in challenges:
        body += f"- {c}\n"
    return md(body)


def write_notebook(filename: str, cells: list[nbf.NotebookNode]) -> str:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    os.makedirs(NB_DIR, exist_ok=True)
    path = os.path.join(NB_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return path


# ---------------------------------------------------------------------------
# Shared synthetic "glyph" dataset — ✕ ◯ ┼ ╱ on 20×20 grids. Self-contained,
# so the course never depends on torchvision (broken on the user's machine).
# ---------------------------------------------------------------------------
GLYPH_DATA = '''
# --- The Glyph dataset: ✕ ◯ ┼ ╱  (self-contained, no torchvision needed) ----
GLYPHS = ["cross", "ring", "plus", "slash"]

def _draw_glyph(cls, size=20, rng=None):
    rng = rng or torch.Generator().manual_seed(0)
    img = torch.zeros(size, size)
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    jx = int(torch.randint(-2, 3, (1,), generator=rng))   # random jitter
    jy = int(torch.randint(-2, 3, (1,), generator=rng))
    c = size // 2
    if cls == 0:    # cross ✕ : two diagonals
        img[((xs - ys).abs() + (jx - jy)).abs() <= 1] = 1.0
        img[((xs + ys - size + 1) + jx + jy).abs() <= 1] = 1.0
    elif cls == 1:  # ring ◯
        r2 = (xs - c - jx) ** 2 + (ys - c - jy) ** 2
        img[(r2 >= 25) & (r2 <= 49)] = 1.0
    elif cls == 2:  # plus ┼
        img[(ys - c - jy).abs() <= 1] = 1.0
        img[(xs - c - jx).abs() <= 1] = 1.0
    else:           # slash ╱ : one diagonal
        img[((xs + ys - size + 1) + jx + jy).abs() <= 1] = 1.0
    img = img + 0.08 * torch.randn(size, size, generator=rng)
    return img.clamp(0, 1)

def make_glyphs(n_per_class=300, size=20, seed=0):
    rng = torch.Generator().manual_seed(seed)
    X = torch.stack([_draw_glyph(c, size, rng) for c in range(4) for _ in range(n_per_class)])
    y = torch.arange(4).repeat_interleave(n_per_class)
    perm = torch.randperm(len(y), generator=rng)
    return X[perm].unsqueeze(1), y[perm]   # (N, 1, 20, 20), (N,)
'''
