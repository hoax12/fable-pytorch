import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loss Landscape", page_icon="🏔️", layout="wide")
st.title("🏔️ Loss Landscape Explorer")
st.caption("Companion to Quests 01 & 04 — the terrain that gradient descent must cross")

tab1, tab2 = st.tabs(["🗺️ Exact terrain (2 knobs)", "🕳️ Slice through a real network"])

# ---------------------------------------------------------------------------
with tab1:
    st.markdown(
        "A model with only two knobs (`y = w·x + b`) has a loss surface we can draw **exactly** — "
        "and we can watch the optimizer walk across it. Change the optimizer and watch the footprints change."
    )
    c = st.columns(4)
    opt_name = c[0].selectbox("optimizer", ["SGD", "SGD + momentum", "Adam"])
    lr = c[1].select_slider("learning rate", [0.001, 0.005, 0.01, 0.05, 0.1, 0.3], value=0.05)
    steps = c[2].slider("steps", 10, 300, 80)
    start = c[3].selectbox("start point", ["(-4, 4)", "(4, 4)", "(-4, -3)"])

    torch.manual_seed(0)
    xd = torch.linspace(-2, 2, 120)
    yd = 1.8 * xd - 0.7 + 0.3 * torch.randn(120)

    def loss_at(w, b):
        return ((w * xd + b - yd) ** 2).mean()

    w0, b0 = eval(start)
    w = torch.tensor(float(w0), requires_grad=True)
    b = torch.tensor(float(b0), requires_grad=True)
    opt = {"SGD": lambda: torch.optim.SGD([w, b], lr=lr),
           "SGD + momentum": lambda: torch.optim.SGD([w, b], lr=lr, momentum=0.9),
           "Adam": lambda: torch.optim.Adam([w, b], lr=lr)}[opt_name]()

    path = [(w.item(), b.item())]
    for _ in range(steps):
        opt.zero_grad()
        loss_at(w, b).backward()
        opt.step()
        path.append((w.item(), b.item()))

    W, B = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing="xy")
    with torch.no_grad():
        Z = ((W[..., None] * xd + B[..., None] - yd) ** 2).mean(-1)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    cs = ax.contourf(W, B, Z.log(), levels=30, cmap="terrain")
    px, py = zip(*path)
    ax.plot(px, py, "r.-", ms=4, lw=1.2, label=f"{opt_name} path")
    ax.plot(px[0], py[0], "ws", ms=9, label="start")
    ax.plot(1.8, -0.7, "k*", ms=16, label="true minimum")
    ax.set_xlabel("w"); ax.set_ylabel("b"); ax.legend(loc="upper right")
    ax.set_title(f"final loss: {loss_at(torch.tensor(px[-1]), torch.tensor(py[-1])).item():.4f}")
    fig.colorbar(cs, label="log loss")
    st.pyplot(fig)
    st.info(
        "💡 **Momentum** overshoots then swings back like a ball with inertia. **Adam** takes a "
        "beeline. Crank the LR to 0.3 with plain SGD and watch it ricochet across the valley."
    )

# ---------------------------------------------------------------------------
with tab2:
    st.markdown(
        "A real network has thousands of knobs — we can't draw a 4,000-dimensional surface. The trick "
        "(from the *loss landscape* papers): pick **two random directions** in parameter space and plot "
        "the loss on that 2-D slice around the trained weights."
    )
    c = st.columns(3)
    train_epochs = c[0].slider("training epochs", 0, 400, 200, 50,
                               help="0 = untrained net. See how training reshapes the local terrain!")
    span = c[1].select_slider("slice span", [0.5, 1.0, 2.0, 4.0], value=1.0)
    seed = c[2].slider("direction seed", 0, 20, 0)

    torch.manual_seed(1)
    Xs = torch.randn(300, 2)
    ys_lab = ((Xs[:, 0] ** 2 + Xs[:, 1] ** 2) < 1.2).long()

    @st.cache_data(show_spinner="training + scanning the slice…")
    def scan(train_epochs, span, seed):
        torch.manual_seed(1)
        net = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU(), nn.Linear(24, 2))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(train_epochs):
            opt.zero_grad()
            F.cross_entropy(net(Xs), ys_lab).backward()
            opt.step()
        theta = [p.detach().clone() for p in net.parameters()]
        g = torch.Generator().manual_seed(seed)
        d1 = [torch.randn(p.shape, generator=g) * p.norm() / max(torch.randn(p.shape).norm(), 1e-8) for p in theta]
        d2 = [torch.randn(p.shape, generator=g) * p.norm() / max(torch.randn(p.shape).norm(), 1e-8) for p in theta]
        n = 41
        alphas = torch.linspace(-span, span, n)
        Z = torch.zeros(n, n)
        with torch.no_grad():
            for i, a in enumerate(alphas):
                for j, bta in enumerate(alphas):
                    for p, t, u, v in zip(net.parameters(), theta, d1, d2):
                        p.copy_(t + a * u + bta * v)
                    Z[j, i] = F.cross_entropy(net(Xs), ys_lab)
            for p, t in zip(net.parameters(), theta):
                p.copy_(t)
        return alphas.numpy(), Z.numpy()

    alphas, Z = scan(train_epochs, span, seed)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cs = ax.contourf(alphas, alphas, np.log(Z + 1e-3), levels=30, cmap="terrain")
    ax.plot(0, 0, "r*", ms=16, label="trained weights (center)")
    ax.legend(); ax.set_xlabel("direction 1"); ax.set_ylabel("direction 2")
    ax.set_title(f"log-loss on a random 2-D slice ({train_epochs} epochs of training)")
    fig.colorbar(cs)
    st.pyplot(fig)
    st.info(
        "💡 Set epochs to **0**: the center is nothing special. At **200+**: the trained weights sit "
        "in a *basin* — training dug a valley. Wider, flatter basins are associated with better "
        "generalization; that's an active research area you can now literally see."
    )
