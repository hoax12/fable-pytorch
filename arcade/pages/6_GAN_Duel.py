import streamlit as st
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="GAN Duel", page_icon="⚔️", layout="wide")
st.title("⚔️ GAN Duel")
st.caption("Companion to Quest 11 — referee a live match between forger and detective")

st.markdown(
    "The **forger** (generator) has never seen the target shape. Its only teacher is the "
    "**detective** (discriminator). Press *Fight* to run 500 duel rounds at a time and watch the "
    "forger's output converge on the target — this state persists between rounds."
)

TARGETS = {
    "heart ♥": lambda n: _heart(n),
    "ring ◯": lambda n: _ring(n),
    "yin-yang blobs": lambda n: _blobs(n),
}

def _heart(n):
    t = torch.rand(n) * 2 * math.pi
    x = 16 * torch.sin(t) ** 3
    y = 13 * torch.cos(t) - 5 * torch.cos(2 * t) - 2 * torch.cos(3 * t) - torch.cos(4 * t)
    return torch.stack([x, y], 1) / 16 + 0.03 * torch.randn(n, 2)

def _ring(n):
    t = torch.rand(n) * 2 * math.pi
    r = 1 + 0.05 * torch.randn(n)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)

def _blobs(n):
    c = torch.tensor([[-0.7, 0.0], [0.7, 0.0]])
    return torch.cat([ci + 0.25 * torch.randn(n // 2, 2) for ci in c])

target_name = st.selectbox("target shape (the forger never sees this directly)", list(TARGETS.keys()))

ss = st.session_state
if ss.get("duel_target") != target_name:
    torch.manual_seed(0)
    ss.duel_target = target_name
    ss.G = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    ss.D = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    ss.optG = torch.optim.Adam(ss.G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    ss.optD = torch.optim.Adam(ss.D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    ss.real = TARGETS[target_name](2000)
    ss.rounds = 0
    ss.lossG_hist, ss.lossD_hist = [], []

c1, c2, c3 = st.columns([1, 1, 2])
fight = c1.button("⚔️ Fight! (+500 rounds)", type="primary", use_container_width=True)
reset = c2.button("🔄 rematch (reset)", use_container_width=True)
c3.metric("rounds fought", ss.rounds)

if reset:
    ss.duel_target = None
    st.rerun()

if fight:
    bce = nn.BCEWithLogitsLoss()
    prog = st.progress(0.0, text="dueling…")
    for i in range(500):
        idx = torch.randint(0, len(ss.real), (256,))
        fake = ss.G(torch.randn(256, 8)).detach()
        lossD = bce(ss.D(ss.real[idx]), torch.ones(256, 1)) + bce(ss.D(fake), torch.zeros(256, 1))
        ss.optD.zero_grad(); lossD.backward(); ss.optD.step()

        fake = ss.G(torch.randn(256, 8))
        lossG = bce(ss.D(fake), torch.ones(256, 1))
        ss.optG.zero_grad(); lossG.backward(); ss.optG.step()

        if i % 25 == 0:
            ss.lossG_hist.append(lossG.item())
            ss.lossD_hist.append(lossD.item())
            prog.progress(i / 500, text=f"round {ss.rounds + i}")
    ss.rounds += 500
    prog.progress(1.0, text="round complete")

with torch.no_grad():
    fake_pts = ss.G(torch.randn(1500, 8))

col_a, col_b = st.columns([3, 2])
with col_a:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ss.real[:, 0], ss.real[:, 1], s=4, alpha=0.15, c="gray", label="target (hidden from forger)")
    ax.scatter(fake_pts[:, 0], fake_pts[:, 1], s=4, alpha=0.5, c="crimson", label="forger's output")
    ax.legend(loc="upper right"); ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.5); ax.axis("off")
    ax.set_title(f"after {ss.rounds} rounds")
    st.pyplot(fig)
with col_b:
    if ss.lossG_hist:
        fig2, ax2 = plt.subplots(figsize=(4.5, 3))
        ax2.plot(ss.lossG_hist, label="forger loss", c="crimson")
        ax2.plot(ss.lossD_hist, label="detective loss", c="steelblue")
        ax2.legend(); ax2.set_xlabel("checkpoints")
        ax2.set_title("the arms race")
        st.pyplot(fig2)
    st.info(
        "💡 Healthy duel: both losses hover — neither side wins outright. If the detective's "
        "loss crashes to ~0, the forger gets no signal (a real failure mode called "
        "*discriminator overpowering*). Watch for **mode collapse** on the two-blob target: "
        "the forger may cover only one blob!"
    )
