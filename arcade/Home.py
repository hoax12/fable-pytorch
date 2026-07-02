"""🕹️ The Arcade — interactive companion to The Tensor Forge.

Run:  streamlit run arcade/Home.py
"""
import streamlit as st

st.set_page_config(page_title="The Tensor Forge Arcade", page_icon="🕹️", layout="wide")

st.title("🕹️ The Tensor Forge — Arcade")
st.markdown(
    "Six machines, one goal: **feel** the ideas from the quests with your hands. "
    "Pick a machine from the sidebar."
)

cols = st.columns(3)
machines = [
    ("🎬 Training Theater", "Watch a decision boundary being carved, epoch by epoch.", "Quests 01–05"),
    ("🏔️ Loss Landscape", "Fly over a real loss surface and trace where descent walked.", "Quests 01, 04"),
    ("🎲 Guess the Gradient", "A game: predict derivatives by hand — autograd is the referee.", "Quests 02, 04"),
    ("🔬 Convolution Lab", "Compose your own 3×3 filter and watch what it detects.", "Quest 07"),
    ("🔍 Attention Microscope", "Zoom into attention heads, causal masks, and score temperature.", "Quest 09"),
    ("⚔️ GAN Duel", "Referee a live match: forger vs detective, 500 steps per round.", "Quest 11"),
]
for i, (name, desc, quest) in enumerate(machines):
    with cols[i % 3]:
        st.subheader(name)
        st.write(desc)
        st.caption(f"companion to {quest}")

st.divider()
import torch
st.caption(f"PyTorch {torch.__version__} · CUDA {'✅' if torch.cuda.is_available() else '❌ (CPU)'} · "
           "All machines run comfortably on CPU.")
