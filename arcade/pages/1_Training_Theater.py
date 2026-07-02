import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Training Theater", page_icon="🎬", layout="wide")
st.title("🎬 Training Theater")
st.caption("Companion to Quests 01–05 — watch learning happen, epoch by epoch")

st.markdown(
    "Train a small network and **scrub through its history** like a film. You'll see the decision "
    "boundary start random, then get carved into shape as the loss falls."
)

with st.sidebar:
    dataset = st.selectbox("dataset", ["spirals", "moons", "blobs", "checker"])
    hidden = st.select_slider("hidden width", [8, 16, 32, 64, 128], value=32)
    depth = st.slider("hidden layers", 1, 4, 2)
    lr = st.select_slider("learning rate", [0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
    epochs = st.slider("epochs", 100, 1500, 600, 100)
    go = st.button("🎬 Film a training run", type="primary", use_container_width=True)


def make_data(kind, n=500):
    g = torch.Generator().manual_seed(0)
    if kind == "spirals":
        n2 = n // 2
        idx = torch.arange(n2).float()
        r, th = idx / n2 * 1.6, idx / n2 * 3 * np.pi
        s0 = torch.stack([r * torch.sin(th), r * torch.cos(th)], 1)
        s1 = torch.stack([r * torch.sin(th + np.pi), r * torch.cos(th + np.pi)], 1)
        X = torch.cat([s0, s1]) + 0.12 * torch.randn(n, 2, generator=g)
        y = torch.cat([torch.zeros(n2), torch.ones(n2)]).long()
    elif kind == "moons":
        n2 = n // 2
        t = torch.linspace(0, np.pi, n2)
        x0 = torch.stack([torch.cos(t), torch.sin(t)], 1)
        x1 = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)
        X = torch.cat([x0, x1]) + 0.15 * torch.randn(n, 2, generator=g)
        y = torch.cat([torch.zeros(n2), torch.ones(n2)]).long()
    elif kind == "blobs":
        c = torch.tensor([[-1.5, -1.], [1.5, -1.], [0., 1.7]])
        X = torch.cat([ci + 0.5 * torch.randn(n // 3, 2, generator=g) for ci in c])
        y = torch.arange(3).repeat_interleave(n // 3)
    else:  # checker
        X = torch.rand(n, 2, generator=g) * 4 - 2
        y = (((X[:, 0] > 0) ^ (X[:, 1] > 0))).long()
    return X, y


X, y = make_data(dataset)
n_classes = int(y.max().item()) + 1

if go:
    torch.manual_seed(3)
    layers, d = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, n_classes))
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # grid for boundary snapshots
    gx = torch.linspace(X[:, 0].min() - .4, X[:, 0].max() + .4, 160)
    gy = torch.linspace(X[:, 1].min() - .4, X[:, 1].max() + .4, 160)
    GX, GY = torch.meshgrid(gx, gy, indexing="xy")
    grid = torch.stack([GX.flatten(), GY.flatten()], 1)

    n_frames = 30
    snap_every = max(1, epochs // n_frames)
    frames, losses, accs = [], [], []
    prog = st.progress(0.0, text="filming…")
    for e in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if e % snap_every == 0 or e == epochs - 1:
            with torch.no_grad():
                probs = F.softmax(model(grid), 1)
                Z = probs[:, min(1, n_classes - 1)] if n_classes == 2 else probs.argmax(1).float()
                frames.append((e, Z.reshape(GX.shape).clone()))
                accs.append((model(X).argmax(1) == y).float().mean().item())
        if e % 50 == 0:
            prog.progress(e / epochs, text=f"epoch {e} · loss {loss.item():.3f}")
    prog.progress(1.0, text="🎬 wrap!")
    st.session_state.theater = dict(frames=frames, losses=losses, accs=accs,
                                    GX=GX, GY=GY, X=X, y=y, n_classes=n_classes)

if "theater" in st.session_state:
    T = st.session_state.theater
    idx = st.slider("⏪ scrub through training ⏩", 0, len(T["frames"]) - 1, len(T["frames"]) - 1)
    epoch_num, Z = T["frames"][idx]

    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.contourf(T["GX"], T["GY"], Z, levels=16, cmap="coolwarm", alpha=0.65)
        ax.scatter(T["X"][:, 0], T["X"][:, 1], c=T["y"], cmap="coolwarm", s=8,
                   edgecolors="k", linewidths=0.2)
        ax.set_title(f"epoch {epoch_num} — train accuracy {T['accs'][idx]*100:.1f}%")
        ax.axis("off")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(4.5, 3))
        ax2.plot(T["losses"], lw=1, color="gray", alpha=0.6)
        ax2.axvline(epoch_num, color="crimson", lw=2)
        ax2.set_xlabel("epoch"); ax2.set_ylabel("loss"); ax2.set_title("you are here")
        st.pyplot(fig2)
        st.metric("frame", f"{idx + 1}/{len(T['frames'])}")

    st.info(
        "💡 Scrub back to frame 1: the boundary is a random smear. Watch it *snap* into place "
        "around the epoch where the loss curve bends. Try **1 layer, width 8** on spirals — "
        "the network is too weak to carve the shape. Depth is capacity."
    )
else:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=8, edgecolors="k", linewidths=0.2)
    ax.set_title(f"{dataset} — press '🎬 Film a training run'")
    ax.axis("off")
    st.pyplot(fig)
