import streamlit as st
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Attention Microscope", page_icon="🔍", layout="wide")
st.title("🔍 Attention Microscope")
st.caption("Companion to Quest 09 — see softmax(QKᵀ/√d) with your own eyes")

st.markdown(
    "Type a sentence and inspect the attention machinery: **two heads**, a causal mask toggle, "
    "and a score-temperature dial. Projections are random (untrained), so you're studying the "
    "*mechanism*, not learned meaning — exactly like holding a lens up to the raw machine."
)

sentence = st.text_input("sentence", "the forge glows hot and the hammer falls")
c = st.columns(4)
d_model = c[0].select_slider("embedding dim", [8, 16, 32, 64], value=16)
causal = c[1].checkbox("causal mask (GPT-style)", value=True)
sharp = c[2].select_slider("score sharpness", [0.25, 0.5, 1.0, 2.0, 4.0], value=1.0,
                           help="multiplies the scores before softmax — like 1/temperature")
seed = c[3].slider("projection seed", 0, 30, 4)

tokens = sentence.split()
T = len(tokens)
if T < 2:
    st.warning("need at least two words")
    st.stop()

torch.manual_seed(seed)

def embed(tok):
    g = torch.Generator().manual_seed(abs(hash(tok)) % (2 ** 31))
    return torch.randn(d_model, generator=g)

X = torch.stack([embed(t) for t in tokens])

fig, axes = plt.subplots(1, 2, figsize=(2.2 + 0.62 * T * 2, 1.8 + 0.55 * T))
for h, ax in enumerate(axes):
    Wq, Wk = torch.randn(d_model, d_model), torch.randn(d_model, d_model)
    Q, K = X @ Wq, X @ Wk
    scores = (Q @ K.T) / math.sqrt(d_model) * sharp
    if causal:
        scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    A = F.softmax(scores, dim=-1)
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(T)); ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(f"head {h + 1}")
fig.colorbar(im, ax=axes, fraction=0.03)
st.pyplot(fig)

st.markdown("#### Reading the map")
st.markdown(
    "- Each **row** is one token's attention budget — it always sums to **1**.\n"
    "- With the **causal mask**, the upper triangle is zero: no peeking at future tokens. "
    "Notice the first token can only attend to itself (its whole row is one cell).\n"
    "- Crank **sharpness** to 4: rows collapse onto single cells (hard pointers). "
    "Drop it to 0.25: rows blur toward uniform (everything attends to everything).\n"
    "- The two heads disagree — different projections find different relationships. "
    "That's why transformers run many heads in parallel."
)

st.info(
    "💡 Repeat a word in your sentence (e.g. 'the ... the') — identical tokens get identical "
    "embeddings here, so their **columns** look alike. In a trained model, *positional* "
    "embeddings break that tie. Quest 09 shows why removing them destroys the model."
)
