# ⚒️ The Tensor Forge — Learn PyTorch by Building It

> 📖 **New here? Read [`HOW_TO_USE.md`](HOW_TO_USE.md) first** — setup, the XP system, pacing, and troubleshooting.

An interactive PyTorch course with a twist: **before you use the magic, you forge it yourself.**

You start by building your own tiny autograd engine from scratch — then meet PyTorch and
discover it's the same machine, industrialized. From there you wield the real framework through
CNNs, Transformers, generative models, and RL, survive a **Debugging Dojo**, and defeat a
**Final Boss** capstone.

## ✨ What makes this course different

| Feature | What it means |
|---------|---------------|
| ⚙️ **Forge-first pedagogy** | You build a working autograd engine in ~60 lines *before* touching `loss.backward()`. Nothing is magic afterwards. |
| 🎯 **Auto-graded exercises + XP** | Every notebook has `check("ex1", answer)` — instant ✅/❌ feedback, hints on failure, and XP that accumulates as you play. |
| 🥋 **Debugging Dojo** | Seven realistically broken training loops (the seven classic PyTorch bugs). Diagnose, fix, get graded. |
| 🕹️ **The Arcade** | A Streamlit app of live demos: watch a net learn, explore a real loss landscape, play *Guess the Gradient*, duel a GAN. |
| 👾 **Final Boss** | A multi-phase capstone with auto-graded milestones. Beat it and you've genuinely learned PyTorch. |

## 🗺️ The Journey

### Act I — Forge the Engine *(understand by building)*
| # | Notebook | The quest |
|---|----------|-----------|
| 01 | [The Idea of Learning](notebooks/01_the_idea_of_learning.ipynb) | What "learning" actually is: knobs, loss, and rolling downhill |
| 02 | [Build Your Own Autograd](notebooks/02_build_your_own_autograd.ipynb) | Forge a `Value` class that computes gradients — then train a real neuron with it |
| 03 | [Tensors — The Real Metal](notebooks/03_tensors_the_real_metal.ipynb) | PyTorch tensors: shapes, broadcasting, and why they're fast |
| 04 | [Autograd Unmasked](notebooks/04_autograd_unmasked.ipynb) | PyTorch's engine, verified against *yours*. The magic is gone (in the best way) |
| 05 | [Your First Real Network](notebooks/05_your_first_real_network.ipynb) | `nn.Module`, optimizers, and the training loop you'll write forever |

### Act II — Wield the Framework
| # | Notebook | The quest |
|---|----------|-----------|
| 06 | [Feeding the Beast](notebooks/06_feeding_the_beast.ipynb) | Datasets, DataLoaders, batching, splits, transforms |
| 07 | [Eyes: Convolutions](notebooks/07_eyes_convolutions.ipynb) | CNNs — train a glyph classifier and dissect what it learned |
| 08 | [Memory: Sequences](notebooks/08_memory_sequences.ipynb) | RNN/GRU/LSTM — forecast signals and generate text char-by-char |
| 09 | [Attention: Build a GPT](notebooks/09_attention_build_a_gpt.ipynb) | Self-attention from first principles → a working mini-GPT |
| 10 | [🥋 The Debugging Dojo](notebooks/10_debugging_dojo.ipynb) | Seven broken training loops. Fix them all. Auto-graded. |

### Act III — Master the Arts
| # | Notebook | The quest |
|---|----------|-----------|
| 11 | [The Art of Creation](notebooks/11_art_of_creation.ipynb) | Autoencoders → VAE → GAN → diffusion, all runnable on CPU |
| 12 | [The Art of Action](notebooks/12_art_of_action.ipynb) | RL: policy gradients and DQN on CartPole |
| 13 | [The Art of Speed](notebooks/13_art_of_speed.ipynb) | Profiling, `torch.compile`, AMP, quantization, export |
| 14 | [👾 The Final Boss](notebooks/14_final_boss.ipynb) | Multi-phase capstone with auto-graded milestones |

### Epilogue — Beyond the Forge
| # | Notebook | The quest |
|---|----------|-----------|
| 15 | [📦 Beyond the Forge — ONNX](notebooks/15_beyond_the_forge_onnx.ipynb) | Export, validate & quantize an ONNX artifact; serve it **torch-free** with ONNX Runtime (see [`ship/`](ship/)) |

### 🕹️ The Arcade (Streamlit app)
```bash
streamlit run arcade/Home.py
```
| Demo | What you do |
|------|-------------|
| 🎬 Training Theater | Watch a network's decision boundary evolve epoch-by-epoch, animated |
| 🏔️ Loss Landscape | Fly over a *real* network's loss surface and see where SGD walked |
| 🎲 Guess the Gradient | A game: predict gradients by hand; autograd grades you |
| 🔬 Convolution Lab | Compose your own 3×3 filter with sliders and see what it detects |
| 🔍 Attention Microscope | Inspect attention head patterns, causal masks, and temperature |
| ⚔️ GAN Duel | Watch a generator and discriminator battle in real time |

## 🚀 Quick start

```powershell
pip install -r requirements.txt

jupyter lab                     # start with notebooks/01_...
streamlit run arcade/Home.py    # keep the Arcade open beside it
```

**On Colab:** push this folder to GitHub, open a notebook, and use `Runtime → GPU`. Every
notebook is CPU-sized by default with clearly marked knobs (`STEPS`, `EPOCHS`) to crank up on GPU.

> ⚠️ **Local note:** your machine's `torchvision` is version-mismatched with torch. All vision
> content here uses a self-contained synthetic **glyph dataset** (`✕ ◯ ┼ ╱`), so nothing breaks —
> and it trains in seconds on CPU.

## 🎯 How the XP system works

Each notebook defines exercises checked by an inline grader:

```python
check("ex1", my_answer)   # ✅ +10 XP  ...or ❌ with a hint, try again
xp_report()               # your score for this notebook
```

No accounts, no files — XP lives in your kernel session. The point isn't the number; it's that
**you can't fool yourself into thinking you understood something you didn't.**

Forge well. ⚒️🔥
