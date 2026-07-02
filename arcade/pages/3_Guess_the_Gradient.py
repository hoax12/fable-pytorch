import streamlit as st
import torch
import random

st.set_page_config(page_title="Guess the Gradient", page_icon="🎲", layout="centered")
st.title("🎲 Guess the Gradient")
st.caption("Companion to Quests 02 & 04 — autograd is the referee")

st.markdown(
    "You get a function and a point. **Compute `df/dx` in your head (or on paper)** and submit. "
    "Autograd checks you. Build a streak. 🔥"
)

PUZZLES = [
    # (display, fn, x-values to sample from)
    ("f(x) = x²", lambda x: x ** 2, [1.0, 2.0, 3.0, -2.0]),
    ("f(x) = x³", lambda x: x ** 3, [1.0, 2.0, -1.0]),
    ("f(x) = 3x² + 2x", lambda x: 3 * x ** 2 + 2 * x, [1.0, 2.0, 0.0, -1.0]),
    ("f(x) = 1/x", lambda x: 1 / x, [1.0, 2.0, 0.5]),
    ("f(x) = sin(x)", lambda x: torch.sin(x), [0.0]),
    ("f(x) = eˣ", lambda x: torch.exp(x), [0.0, 1.0]),
    ("f(x) = x · sin(x)", lambda x: x * torch.sin(x), [0.0]),
    ("f(x) = (2x + 1)²", lambda x: (2 * x + 1) ** 2, [0.0, 1.0, -1.0]),
    ("f(x) = tanh(x)", lambda x: torch.tanh(x), [0.0]),
    ("f(x) = x² · eˣ", lambda x: x ** 2 * torch.exp(x), [0.0, 1.0]),
    ("f(x) = ln(x)", lambda x: torch.log(x), [1.0, 2.0]),
    ("f(x) = √x", lambda x: torch.sqrt(x), [1.0, 4.0]),
]

ss = st.session_state
if "gg_puzzle" not in ss:
    ss.gg_streak = 0
    ss.gg_best = 0
    ss.gg_solved = 0
    ss.gg_puzzle = None

def new_puzzle():
    i = random.randrange(len(PUZZLES))
    disp, fn, xs = PUZZLES[i]
    ss.gg_puzzle = (disp, i, random.choice(xs))
    ss.gg_revealed = False

if ss.gg_puzzle is None:
    new_puzzle()

disp, idx, x_val = ss.gg_puzzle
_, fn, _ = PUZZLES[idx]

c1, c2, c3 = st.columns(3)
c1.metric("🔥 streak", ss.gg_streak)
c2.metric("🏆 best streak", ss.gg_best)
c3.metric("✅ solved", ss.gg_solved)

st.divider()
st.markdown(f"## {disp}")
st.markdown(f"### What is  df/dx  at  **x = {x_val:g}** ?")

guess = st.number_input("your answer", value=0.0, step=0.1, format="%.4f")

col_a, col_b = st.columns(2)
if col_a.button("⚖️ Judge me", type="primary", use_container_width=True):
    x = torch.tensor(float(x_val), requires_grad=True)
    y = fn(x)
    y.backward()
    truth = x.grad.item()
    if abs(guess - truth) < max(0.02, abs(truth) * 0.02):
        ss.gg_streak += 1
        ss.gg_solved += 1
        ss.gg_best = max(ss.gg_best, ss.gg_streak)
        st.success(f"✅ Correct! autograd says {truth:.4f}. Streak: {ss.gg_streak} 🔥")
        new_puzzle()
        st.rerun()
    else:
        ss.gg_streak = 0
        st.error(f"❌ autograd says **{truth:.4f}** — you said {guess:.4f}. Streak reset!")

if col_b.button("⏭️ skip / new puzzle", use_container_width=True):
    new_puzzle()
    st.rerun()

with st.expander("🧠 cheat sheet (no shame — memorize through use)"):
    st.markdown(
        """
| f(x) | f'(x) |
|------|-------|
| xⁿ | n·xⁿ⁻¹ |
| eˣ | eˣ |
| ln x | 1/x |
| sin x | cos x |
| tanh x | 1 − tanh²x |
| g(h(x)) | g'(h(x)) · h'(x) — **the chain rule your Value class implements** |
| u·v | u'v + uv' |
"""
    )
