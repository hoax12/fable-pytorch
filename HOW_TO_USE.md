# 📖 How to Use The Tensor Forge

Read this once before Quest 01. It covers how to run everything, how the XP system works, and
what to do when something breaks.

---

## 1. The map

```
fable_folder/
├── README.md                  # curriculum overview
├── HOW_TO_USE.md              # ← you are here
├── requirements.txt
├── notebooks/                 # ⚒️ the 15 quests, in order
│   ├── 01–05  Act I    — forge the engine (build autograd yourself!)
│   ├── 06–10  Act II   — wield the framework (ends in the Debugging Dojo 🥋)
│   ├── 11–14  Act III  — master the arts (ends at the Final Boss 👾)
│   └── 15     Epilogue — escape the forge with ONNX
├── arcade/                    # 🕹️ Streamlit demos (Home.py + 6 machines)
├── ship/                      # 🚀 torch-free deployment artifacts (Quest 15)
└── tools/                     # notebook generators — you can ignore these
```

---

## 2. Running it

### Locally (everything works on CPU)

```powershell
# from fable_folder/
pip install -r requirements.txt

jupyter lab                    # open notebooks/01_the_idea_of_learning.ipynb
streamlit run arcade/Home.py   # in a second terminal — keep the Arcade beside you
```

Every quest runs in seconds-to-minutes on CPU. No dataset downloads, ever — the course uses
self-contained synthetic data (the glyphs ✕ ◯ ┼ ╱, spirals, hearts ♥), so it also works offline
and sidesteps your machine's broken `torchvision` entirely.

### On Google Colab (for cranking up scale)

1. Push this folder to GitHub.
2. Point the badges at your repo and rebuild (only needed if you fork or rename):
   ```powershell
   python tools/set_colab_repo.py hoax12/fable-pytorch main
   ```
   Default badges already target `github.com/hoax12/fable-pytorch` → `notebooks/` at repo root.
3. Open any notebook via its **Open in Colab** badge → `Runtime → Change runtime type → GPU`.
4. Find the scale knobs (`STEPS`, `EPOCHS` — always marked with 🔼) and crank them.

---

## 3. How to play a quest

Each quest follows the same rhythm:

1. **Read** the idea (short — the code carries the weight).
2. **Run** every cell in order. The plots are the payoff; look at them.
3. **Break things.** Change a number, rerun, watch what happens. This is 50% of the value.
4. **Fight the Boss Challenges** at the end:

```python
check("ex_name", your_answer)   # instant ✅ (+XP) or ❌ (+hint) — retry freely
xp_report()                     # progress bar for the quest
```

The grader lives *inside* the notebook (nothing to install, works on Colab). XP resets with the
kernel — the score is a self-honesty device, not a leaderboard. **You cannot fool yourself into
thinking you understood something you didn't.**

> Solutions: quests 10 (Dojo) and 14 (Boss) include full "sensei" walkthroughs at the bottom.
> For other quests, the hints in each `check` failure are designed to walk you to the answer.

### Pair each quest with an Arcade machine

| After quest | Open |
|-------------|------|
| 01–02 | 🎲 Guess the Gradient, 🏔️ Loss Landscape |
| 04–05 | 🎬 Training Theater |
| 07 | 🔬 Convolution Lab |
| 09 | 🔍 Attention Microscope |
| 11 | ⚔️ GAN Duel |

---

## 4. Suggested pacing

| Plan | Schedule |
|------|----------|
| 🏃 Intense weekend | Day 1: Act I + Dojo prep · Day 2: Acts II–III highlights |
| 🚶 Two weeks (recommended) | one quest per day; Dojo and Boss get a full day each |
| 🧗 Mastery | one quest per 2 days; earn **every** XP point; redo the Dojo from memory a week later |

Milestones worth celebrating:
- ⚙️ Quest 02 — you built backprop *yourself*.
- 🧠 Quest 09 — you built a GPT and understand every line.
- 🥋 Quest 10 — all seven stations cleared = debugging black belt.
- 👾 Quest 14 — boss defeated = you can ship a model end-to-end.

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `check` says ❌ but you're sure you're right | Read the hint carefully — checks validate *exact* types/shapes (e.g. tensor vs float, tuple vs list). |
| `unknown exercise` from `check(...)` | Run the quest's **registry cell** (the one full of `_register(...)` calls). |
| Kernel restarted, XP gone | Expected — XP is per-session. Re-run setup + registry cells and re-check; correct answers re-earn instantly. |
| Training slower than the printed expectations | You're on a modest CPU — lower `STEPS`/`EPOCHS`, or move to Colab GPU. |
| `torchvision` errors anywhere | You won't hit any in this course (it never imports torchvision). If you see one, you're in the other course 😄. |
| `gymnasium`/`onnx` missing | Quests 12 and 15 auto-fallback or auto-install. To pre-install: `pip install gymnasium onnx onnxruntime`. |
| Arcade page crashes | Run from `fable_folder/`: `streamlit run arcade/Home.py`. |
| Windows `DataLoader` hangs | Keep `num_workers=0` in notebooks (the course default). |

---

## 6. After the epilogue

- Re-fight the **Final Boss** on a dataset you actually care about — same five phases.
- Take your Quest 09 GPT to Colab, feed it a real book, train 5,000+ steps.
- Serve the Quest 15 oracle from **FastAPI**, or run the `.onnx` in a browser with ONNX Runtime Web.
- Read [karpathy/micrograd](https://github.com/karpathy/micrograd) and smile — you built it in Quest 02.

⚒️ *Forge well.*
