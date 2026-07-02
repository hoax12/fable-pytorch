import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Convolution Lab", page_icon="🔬", layout="wide")
st.title("🔬 Convolution Lab")
st.caption("Companion to Quest 07 — build a filter with your own hands")

st.markdown(
    "Set the nine numbers of a 3×3 filter yourself and see **exactly** what it detects on the "
    "course glyphs. This is what `Conv2d` learns to do automatically — you're doing it manually."
)

GLYPHS = ["cross ✕", "ring ◯", "plus ┼", "slash ╱"]

def draw_glyph(cls, size=20):
    img = torch.zeros(size, size)
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    c = size // 2
    if cls == 0:
        img[(xs - ys).abs() <= 1] = 1.0
        img[(xs + ys - size + 1).abs() <= 1] = 1.0
    elif cls == 1:
        r2 = (xs - c) ** 2 + (ys - c) ** 2
        img[(r2 >= 25) & (r2 <= 49)] = 1.0
    elif cls == 2:
        img[(ys - c).abs() <= 1] = 1.0
        img[(xs - c).abs() <= 1] = 1.0
    else:
        img[(xs + ys - size + 1).abs() <= 1] = 1.0
    return img

PRESETS = {
    "— custom —": None,
    "↘ diagonal detector": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
    "↙ anti-diagonal": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
    "─ horizontal bar": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
    "│ vertical bar": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
    "outline (Laplacian)": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    "blur": [[1/9, 1/9, 1/9]] * 3,
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}

c_left, c_right = st.columns([1, 2])
with c_left:
    glyph_idx = st.radio("test image", range(4), format_func=lambda i: GLYPHS[i], horizontal=True)
    preset = st.selectbox("filter preset", list(PRESETS.keys()), index=1)
    apply_relu = st.checkbox("ReLU after (like a real CNN)", value=True)

    default = PRESETS[preset] or [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    st.write("**the nine numbers:**")
    kernel = []
    for r in range(3):
        cols = st.columns(3)
        row = [cols[cc].number_input(f"k{r}{cc}", value=float(default[r][cc]),
                                     step=0.5, format="%.2f", label_visibility="collapsed")
               for cc in range(3)]
        kernel.append(row)
    K = torch.tensor(kernel, dtype=torch.float32)

img = draw_glyph(glyph_idx)
out = F.conv2d(img.reshape(1, 1, 20, 20), K.reshape(1, 1, 3, 3), padding=1)[0, 0]
if apply_relu:
    out = F.relu(out)

with c_right:
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("input glyph")
    im1 = ax[1].imshow(K, cmap="RdBu", vmin=-max(2, K.abs().max()), vmax=max(2, K.abs().max()))
    ax[1].set_title("your filter")
    for (i, j), v in np.ndenumerate(K.numpy()):
        ax[1].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10)
    ax[2].imshow(out, cmap="inferno"); ax[2].set_title("feature map" + (" (after ReLU)" if apply_relu else ""))
    for a in ax:
        a.axis("off")
    st.pyplot(fig)

    total = out.sum().item()
    st.metric("total activation on this glyph", f"{total:.1f}",
              help="a good detector fires strongly on its target pattern and weakly elsewhere")

st.divider()
st.markdown("#### 🔎 Filter response profile — does your filter *discriminate*?")
scores = []
for g in range(4):
    o = F.conv2d(draw_glyph(g).reshape(1, 1, 20, 20), K.reshape(1, 1, 3, 3), padding=1)[0, 0]
    if apply_relu:
        o = F.relu(o)
    scores.append(o.sum().item())
fig2, ax2 = plt.subplots(figsize=(7, 2.2))
bars = ax2.bar(GLYPHS, scores, color=["crimson" if i == glyph_idx else "steelblue" for i in range(4)])
ax2.set_ylabel("total activation")
st.pyplot(fig2)

st.info(
    "💡 **Challenge:** build a filter that fires strongly on the **plus ┼** but weakly on the "
    "**cross ✕**. (Hint: plus is made of horizontal + vertical strokes; cross is diagonals.) "
    "A CNN's first layer is a *bank* of ~16 such filters, discovered by gradient descent."
)
