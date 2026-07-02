"""Act II — Wield the Framework (quests 06–10)."""
from nb_utils import (header, setup_cell, registry_cell, md, code, boss_md,
                      write_notebook, GLYPH_DATA)


def build():
    _q06()
    _q07()
    _q08()
    _q09()
    _q10()


# ---------------------------------------------------------------------------
# 06 — Feeding the Beast
# ---------------------------------------------------------------------------
def _q06():
    cells = [
        header("06_feeding_the_beast.ipynb", "Act II", "06", "Feeding the Beast",
               "Models are only as good as their food: Datasets, DataLoaders, splits, and augmentation.",
               prev="05_your_first_real_network.ipynb", nxt="07_eyes_convolutions.ipynb"),
        setup_cell(),
        md('''
## Why not just `model(all_the_data)`?

Three reasons real training feeds **mini-batches**:
1. **Memory** — datasets don't fit in RAM/VRAM.
2. **Speed** — more parameter updates per pass over the data.
3. **Noise is good** — slightly noisy gradients help escape bad valleys.

PyTorch splits the job cleanly:
- **`Dataset`** — "how do I fetch sample *i*?" (`__len__` + `__getitem__`)
- **`DataLoader`** — batching, shuffling, parallel workers, on top of any Dataset.
'''),
        md("### Meet the course mascots: the Glyphs ✕ ◯ ┼ ╱"),
        code(GLYPH_DATA),
        code('''
X, y = make_glyphs(n_per_class=300)
print("images:", X.shape, "| labels:", y.shape, "| classes:", GLYPHS)

fig, ax = plt.subplots(2, 8, figsize=(12, 3.2))
for i in range(16):
    ax[i // 8, i % 8].imshow(X[i, 0], cmap="gray")
    ax[i // 8, i % 8].set_title(GLYPHS[y[i]], fontsize=8)
    ax[i // 8, i % 8].axis("off")
plt.suptitle("The Glyph dataset — jittered, noisy, self-contained"); plt.show()
'''),
        md("### Wrap it in a `Dataset`, feed it through a `DataLoader`"),
        code('''
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

ds = TensorDataset(X, y)              # quickest wrapper for in-memory tensors
print("len:", len(ds), "| sample 0 shapes:", ds[0][0].shape, ds[0][1])

loader = DataLoader(ds, batch_size=64, shuffle=True)
xb, yb = next(iter(loader))
print("one batch:", xb.shape, yb.shape, "| batches per epoch:", len(loader))
'''),
        md('''
### The sacred split

You **must** hold out data the model never trains on — otherwise you're grading a student on
questions they memorized. `random_split` does it in one line.
'''),
        code('''
train_ds, val_ds, test_ds = random_split(ds, [900, 150, 150],
                                         generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)
print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")
print("train: learn | val: tune choices | test: touch ONCE at the very end")
'''),
        md("### Writing your own `Dataset` — the pattern you'll reuse forever"),
        code('''
class GlyphDataset(Dataset):
    """A Dataset that generates glyphs on the fly, with optional augmentation."""
    def __init__(self, n=1200, augment=False, seed=0):
        self.X, self.y = make_glyphs(n // 4, seed=seed)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img, label = self.X[i], self.y[i]
        if self.augment:
            if random.random() < 0.5:
                img = img.flip(-1)                          # mirror
            img = img.roll(random.randint(-2, 2), dims=-1)   # small shift
            img = (img + 0.05 * torch.randn_like(img)).clamp(0, 1)
        return img, label

aug = GlyphDataset(augment=True)
fig, ax = plt.subplots(1, 6, figsize=(9, 1.8))
for i in range(6):
    img, lbl = aug[0]           # same index -> different augmented views!
    ax[i].imshow(img[0], cmap="gray"); ax[i].axis("off")
plt.suptitle("Augmentation: one sample, many views (free extra data)"); plt.show()
'''),
        md('''
### Does batch size matter? See for yourself
'''),
        code('''
import time

def one_epoch(batch_size):
    model = nn.Sequential(nn.Flatten(), nn.Linear(400, 64), nn.ReLU(), nn.Linear(64, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    t0 = time.time()
    for xb, yb in loader:
        opt.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
    return (time.time() - t0) * 1000, len(loader)

for bs in [4, 32, 256]:
    ms, nb = one_epoch(bs)
    print(f"batch_size {bs:4d}: {nb:4d} updates/epoch, {ms:7.1f} ms/epoch")
print("\\nsmall batches = many noisy updates; big batches = few smooth ones. 32–256 is the usual sweet spot.")
'''),
        md('''
> 🧵 **`num_workers`**: on Linux/Colab, `DataLoader(..., num_workers=2)` loads batches in
> parallel processes. On Windows notebooks keep it `0` (multiprocessing + notebooks don't mix well).
'''),
        registry_cell('''
_register("warmup", 5,
    lambda n: n == 19,
    "a Dataset with 1234 samples and batch_size=64, drop_last=False -> how many batches? ceil(1234/64)")
_register("own_dataset", 20,
    lambda d: (lambda s: hasattr(d, "__len__") and len(d) == 100
               and torch.is_tensor(s[0]) and s[0].shape == (3,)
               and torch.is_tensor(s[1]) and s[1].item() == int(s[0].sum().item() > 0))(d[0]),
    "Dataset of 100 samples: x = a (3,) float tensor, y = tensor(1) if x.sum() > 0 else tensor(0)")
_register("split_sizes", 10,
    lambda sizes: list(sizes) == [800, 100, 100],
    "an 80/10/10 split of 1000 samples — submit [train, val, test] sizes")
_register("aug_effect", 10,
    lambda s: s.strip().lower() in ("overfitting", "overfit"),
    "one word: augmentation mainly fights ______")
'''),
        code('''
check("warmup", 19)
'''),
        boss_md([
            '`own_dataset` (20 XP) — write a `Dataset` of 100 samples where `x` is a random `(3,)` tensor and `y = 1 if x.sum() > 0 else 0` (as a tensor); submit an instance.',
            '`split_sizes` (10 XP) — split sizes for 80/10/10 of 1000 samples.',
            '`aug_effect` (10 XP) — one word: what does augmentation mainly fight?',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("06_feeding_the_beast.ipynb", cells)


# ---------------------------------------------------------------------------
# 07 — Eyes: Convolutions
# ---------------------------------------------------------------------------
def _q07():
    cells = [
        header("07_eyes_convolutions.ipynb", "Act II", "07", "Eyes — Convolutions",
               "Give your network vision: filters, feature maps, and a trained glyph classifier you can dissect.",
               prev="06_feeding_the_beast.ipynb", nxt="08_memory_sequences.ipynb"),
        setup_cell(),
        md('''
## Why a `Linear` layer is bad at seeing

Flatten a 20×20 glyph into 400 numbers and a `Linear` layer treats pixel (3,3) and pixel (3,4)
as *totally unrelated* inputs. Move the glyph one pixel and every weight is wrong.

A **convolution** slides one small filter across the whole image:
- **locality** — nearby pixels are processed together,
- **weight sharing** — the same detector works at every position (shift a glyph → same response, shifted),
- **efficiency** — a 3×3 filter is 9 weights, not 400.
'''),
        md("### A filter is a pattern-detector. Watch one hunt for diagonals:"),
        code(GLYPH_DATA),
        code('''
X, y = make_glyphs(n_per_class=200)

# Hand-forged filters
diag = torch.tensor([[2., -1, -1], [-1, 2., -1], [-1, -1, 2.]])     # ↘ detector
anti = diag.flip(1)                                                  # ↙ detector
hbar = torch.tensor([[-1., -1, -1], [2., 2., 2.], [-1., -1, -1]])   # ─ detector
bank = torch.stack([diag, anti, hbar]).unsqueeze(1)                  # (3,1,3,3)

sample = X[y == 0][:1]     # a cross ✕ (made of two diagonals)
maps = F.conv2d(sample, bank, padding=1)

fig, ax = plt.subplots(1, 4, figsize=(11, 3))
ax[0].imshow(sample[0, 0], cmap="gray"); ax[0].set_title("input: cross ✕")
for i, name in enumerate(["↘ detector", "↙ detector", "─ detector"]):
    ax[i + 1].imshow(maps[0, i], cmap="inferno"); ax[i + 1].set_title(name)
for a in ax: a.axis("off")
plt.show()
print("the ↘ and ↙ maps light up along the arms; the ─ map mostly doesn't. Detection!")
'''),
        md('''
### Conv arithmetic — the formula you'll compute in your head forever

For input size \\(N\\), kernel \\(K\\), padding \\(P\\), stride \\(S\\):

\\[ \\text{out} = \\left\\lfloor \\frac{N + 2P - K}{S} \\right\\rfloor + 1 \\]

And parameters of `Conv2d(c_in, c_out, K)` = \\(c_{out} \\cdot (c_{in} \\cdot K \\cdot K + 1)\\).
'''),
        code('''
for (cin, cout, k, p, s) in [(1, 16, 3, 1, 1), (16, 32, 3, 1, 2), (1, 8, 5, 0, 1)]:
    conv = nn.Conv2d(cin, cout, k, padding=p, stride=s)
    out = conv(torch.randn(1, cin, 20, 20))
    n_params = sum(pp.numel() for pp in conv.parameters())
    print(f"Conv2d({cin:2d},{cout:2d},k={k},p={p},s={s}): 20 -> {out.shape[-1]:2d}   params={n_params}")
'''),
        md("### Build & train GlyphNet"),
        code('''
from torch.utils.data import TensorDataset, DataLoader, random_split

ds = TensorDataset(X, y)
train_ds, test_ds = random_split(ds, [640, 160], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

class GlyphNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 20 -> 10
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 10 -> 5
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, n_classes))
    def forward(self, x):
        return self.head(self.features(x))

model = GlyphNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(loader):
    model.eval(); hits = n = 0
    with torch.no_grad():
        for xb, yb in loader:
            hits += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item(); n += len(yb)
    return hits / n

for epoch in range(4):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()
    print(f"epoch {epoch}: test accuracy {accuracy(test_loader)*100:.1f}%")
'''),
        md("### Dissect it: what did the filters *become*?"),
        code('''
# 1) learned first-layer filters
W = model.features[0].weight.detach().cpu()
fig, ax = plt.subplots(2, 8, figsize=(12, 3))
for i in range(16):
    ax[i // 8, i % 8].imshow(W[i, 0], cmap="RdBu"); ax[i // 8, i % 8].axis("off")
plt.suptitle("Learned filters — gradient descent invented its own edge detectors"); plt.show()

# 2) feature maps for one glyph
sample = X[y == 1][:1].to(device)       # a ring ◯
with torch.no_grad():
    fmaps = model.features[0](sample).cpu()[0]
fig, ax = plt.subplots(2, 8, figsize=(12, 3))
for i in range(16):
    ax[i // 8, i % 8].imshow(fmaps[i], cmap="inferno"); ax[i // 8, i % 8].axis("off")
plt.suptitle("Feature maps: 16 different 'views' of one ring"); plt.show()
'''),
        code('''
# 3) confusion matrix — WHERE does it fail?
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb.to(device)).argmax(1).cpu()); trues.append(yb)
preds, trues = torch.cat(preds), torch.cat(trues)
cm = torch.zeros(4, 4, dtype=torch.int)
for t, p in zip(trues, preds):
    cm[t, p] += 1
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(GLYPHS); ax.set_yticks(range(4)); ax.set_yticklabels(GLYPHS)
for i in range(4):
    for j in range(4):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() // 2 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("confusion matrix")
plt.show()
print("note: cross ✕ vs slash ╱ share a diagonal — the most confusable pair. Errors make sense!")
'''),
        registry_cell('''
_register("warmup", 5,
    lambda n: n == 10,
    "N=20, K=3, P=1, S=2 -> floor((20 + 2 - 3)/2) + 1 = 10")
_register("param_count", 10,
    lambda n: n == 4640,
    "Conv2d(16, 32, 3): 32 * (16*3*3 + 1) = 4640")
_register("glyph_92", 20,
    lambda m: isinstance(m, nn.Module) and (lambda: (
        [m.eval()] and sum((m(xb.to(device)).argmax(1).cpu() == yb).sum().item() for xb, yb in test_loader)
        / sum(len(yb) for _, yb in test_loader) >= 0.92))(),
    "reach >= 92% test accuracy — train longer, add a conv block, or augment. Submit the model.")
_register("why_share", 10,
    lambda s: "shift" in s.lower() or "translation" in s.lower() or "position" in s.lower() or "anywhere" in s.lower(),
    "one sentence: weight sharing means the same detector works regardless of ____ in the image")
'''),
        code('''
check("warmup", 10)
'''),
        boss_md([
            '`param_count` (10 XP) — how many parameters in `Conv2d(16, 32, 3)`? Compute by hand first.',
            '`glyph_92` (20 XP) — push test accuracy to ≥92% and submit the model.',
            '`why_share` (10 XP) — in one sentence: why is weight sharing the right idea for images?',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("07_eyes_convolutions.ipynb", cells)


# ---------------------------------------------------------------------------
# 08 — Memory: Sequences
# ---------------------------------------------------------------------------
def _q08():
    cells = [
        header("08_memory_sequences.ipynb", "Act II", "08", "Memory — Sequences",
               "Order matters: recurrent networks that forecast signals and write text one character at a time.",
               prev="07_eyes_convolutions.ipynb", nxt="09_attention_build_a_gpt.ipynb"),
        setup_cell(),
        md('''
## A new kind of input

Images arrive all at once. **Sequences** — sensor readings, text, audio — arrive in order, and
the *order is the information*. A recurrent network carries a hidden state `h`, its running
memory of everything seen so far:

\\[ h_t = f(x_t, h_{t-1}) \\]

`nn.GRU` and `nn.LSTM` add *gates* — learned valves controlling what to remember and forget.
'''),
        md("### Part A — forecast a damped, drifting oscillation"),
        code('''
t = torch.linspace(0, 60, 1200)
signal = torch.exp(-t / 40) * torch.sin(t) + 0.02 * t + 0.05 * torch.randn_like(t)
plt.figure(figsize=(10, 2.5)); plt.plot(t, signal, lw=1)
plt.title("the signal: decaying oscillation + upward drift + noise"); plt.show()

# windows: last 30 points -> next point
W = 30
Xseq = torch.stack([signal[i:i + W] for i in range(len(signal) - W)]).unsqueeze(-1)
Yseq = signal[W:].unsqueeze(-1)
print("windows:", Xseq.shape, "targets:", Yseq.shape)
'''),
        code('''
class Forecaster(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, h = self.gru(x)          # out: (B, T, hidden) — hidden state at EVERY step
        return self.head(out[:, -1])  # predict from the last step's memory

model = Forecaster().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
Xd, Yd = Xseq.to(device), Yseq.to(device)
for epoch in range(80):
    opt.zero_grad()
    loss = F.mse_loss(model(Xd), Yd)
    loss.backward(); opt.step()
print(f"final MSE: {loss.item():.5f}")
'''),
        code('''
# Roll the model forward on its OWN predictions (autoregressive)
model.eval()
window = signal[:W].tolist()
preds = []
with torch.no_grad():
    for _ in range(400):
        x = torch.tensor(window[-W:]).reshape(1, W, 1).to(device)
        nxt = model(x).item()
        preds.append(nxt); window.append(nxt)

plt.figure(figsize=(10, 2.8))
plt.plot(signal[:W + 400], label="truth", alpha=0.6)
plt.plot(range(W, W + 400), preds, "--", label="model, feeding itself")
plt.axvline(W, c="gray", ls=":"); plt.legend(); plt.title("autoregressive forecast"); plt.show()
'''),
        md('''
Notice how errors slowly compound — each prediction becomes the next input. This *autoregressive*
loop is exactly how GPT generates text, one token at a time.

### Part B — a network that writes

Character-level language modeling: read characters, predict the **next** one. Our corpus:
tongue twisters (dense letter patterns, tiny vocabulary, fast to learn).
'''),
        code('''
corpus = (
    "she sells sea shells by the sea shore "
    "peter piper picked a peck of pickled peppers "
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood "
    "fuzzy wuzzy was a bear fuzzy wuzzy had no hair "
) * 25

chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in corpus])
V = len(chars)
print(f"vocab: {V} chars | corpus: {len(data)} chars")
'''),
        code('''
SEQ = 32
def batch(bs=48):
    ix = torch.randint(0, len(data) - SEQ - 1, (bs,))
    xs = torch.stack([data[i:i + SEQ] for i in ix])
    ys = torch.stack([data[i + 1:i + SEQ + 1] for i in ix])   # shifted by one: next-char targets
    return xs.to(device), ys.to(device)

class Scribe(nn.Module):
    def __init__(self, V, emb=24, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(V, emb)         # char id -> learned vector
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)
    def forward(self, x, h=None):
        out, h = self.gru(self.emb(x), h)
        return self.head(out), h

scribe = Scribe(V).to(device)
opt = torch.optim.Adam(scribe.parameters(), lr=3e-3)
for step in range(500):
    xs, ys = batch()
    logits, _ = scribe(xs)
    loss = F.cross_entropy(logits.reshape(-1, V), ys.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 125 == 0:
        print(f"step {step:4d}: loss {loss.item():.3f}")
'''),
        code('''
@torch.no_grad()
def write(prompt="she ", n=150, temperature=0.7):
    scribe.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]]).to(device)
    out = list(prompt)
    logits, h = scribe(idx)
    for _ in range(n):
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        out.append(itos[nxt.item()])
        logits, h = scribe(nxt, h)
    return "".join(out)

for temp in [0.3, 0.8, 1.5]:
    print(f"--- temperature {temp} ---")
    print(write(temperature=temp), "\\n")
'''),
        md('''
**Temperature** divides the logits before softmax: low → confident and repetitive, high → chaotic.
The model learned spelling and word fragments from *nothing but next-char prediction* — the same
objective, scaled a billion-fold, produces ChatGPT. Next quest: the architecture that scaled.
'''),
        registry_cell('''
_register("warmup", 5,
    lambda shp: tuple(shp) == (8, 15, 32),
    "nn.GRU(input_size=4, hidden_size=32, batch_first=True) on input (8, 15, 4): out.shape?")
_register("forecast_close", 15,
    lambda mse: mse < 0.003,
    "train the Forecaster longer (or bigger hidden) until MSE < 0.003; submit loss.item()")
_register("onehot", 15,
    lambda f: torch.equal(f(torch.tensor([1, 0, 2]), 4),
                          torch.tensor([[0., 1, 0, 0], [1., 0, 0, 0], [0., 0, 1, 0]])),
    "def onehot(idx, n): return F.one_hot(idx, n).float()  — or build with zeros + scatter")
_register("temp_quiz", 10,
    lambda s: "repet" in s.lower() or "safe" in s.lower() or "confident" in s.lower() or "determinist" in s.lower(),
    "one phrase: text at very LOW temperature becomes ______")
'''),
        code('''
gru = nn.GRU(input_size=4, hidden_size=32, batch_first=True)
out, h = gru(torch.randn(8, 15, 4))
check("warmup", out.shape)
'''),
        boss_md([
            '`forecast_close` (15 XP) — get the forecaster MSE under `0.003`; submit the final loss value.',
            '`onehot` (15 XP) — write `onehot(idx, n)` producing float one-hot rows; submit the function.',
            '`temp_quiz` (10 XP) — what does very low temperature do to generated text?',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("08_memory_sequences.ipynb", cells)


# ---------------------------------------------------------------------------
# 09 — Attention: Build a GPT
# ---------------------------------------------------------------------------
def _q09():
    cells = [
        header("09_attention_build_a_gpt.ipynb", "Act II", "09", "Attention — Build a GPT",
               "The architecture that ate the world, from a single dot product to a working mini-GPT.",
               prev="08_memory_sequences.ipynb", nxt="10_debugging_dojo.ipynb"),
        setup_cell(),
        md('''
## The problem with memory

Your GRU squeezed *everything it read* into one fixed-size hidden vector. Ask it about the first
word of a long paragraph and the memory has faded. **Attention** removes the bottleneck: every
token gets to *look directly at every other token* and take what it needs.

Each token emits three vectors from learned projections:
- **Q**uery — "what am I looking for?"
- **K**ey — "what do I contain?"
- **V**alue — "what do I hand over if you attend to me?"

\\[ \\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V \\]

### First, feel it with actual numbers
'''),
        code('''
# 3 tokens, 4 dims. Token 0's query happens to match token 2's key.
Q = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0], [0., 0, 1, 0]])
K = torch.tensor([[0., 1, 0, 0], [0., 0, 1, 0], [1., 0, 0, 0]])
V = torch.tensor([[10., 0, 0, 0], [0., 20, 0, 0], [0., 0, 30, 0]])

scores = Q @ K.T / math.sqrt(4)
weights = F.softmax(scores, dim=-1)
out = weights @ V

print("attention weights (rows sum to 1):\\n", weights.round(decimals=2))
print("\\ntoken 0's output:", out[0].round(decimals=1), " <- mostly token 2's value (its key matched!)")
'''),
        md('''
### The causal mask — no peeking at the future

A language model predicts token *t+1* from tokens *≤ t*. Zero out (well, `-inf` out) the upper
triangle of the score matrix and each position can only attend backwards.
'''),
        code('''
T = 6
scores = torch.randn(T, T)
mask = torch.tril(torch.ones(T, T))
weights = F.softmax(scores.masked_fill(mask == 0, float("-inf")), dim=-1)

plt.figure(figsize=(4, 3.5))
plt.imshow(weights, cmap="viridis"); plt.colorbar()
plt.title("causal attention: strictly lower-triangular"); plt.xlabel("attends to"); plt.ylabel("token")
plt.show()
'''),
        md("### Multi-head attention — several conversations at once"),
        code('''
class MultiHeadAttention(nn.Module):
    """h parallel attention heads, each looking for something different."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # split channels into heads: (B, T, C) -> (B, h, T, dk)
        q, k, v = (z.reshape(B, T, self.h, self.dk).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        causal = torch.tril(torch.ones(T, T, device=x.device))
        att = F.softmax(att.masked_fill(causal == 0, float("-inf")), dim=-1)
        z = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(z)

print("shape check:", MultiHeadAttention(64, 4)(torch.randn(2, 10, 64)).shape)
'''),
        md('''
### The Transformer block: attend, then think

`x + attention(norm(x))` then `x + mlp(norm(x))`. The `x +` **residual connections** are what
let 100-layer stacks train at all — gradients always have a highway back.
'''),
        code('''
class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # communicate
        x = x + self.mlp(self.norm2(x))     # compute
        return x

class MiniGPT(nn.Module):
    def __init__(self, V, d_model=96, n_heads=4, n_layers=3, block_size=48):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(V, d_model)
        self.pos = nn.Embedding(block_size, d_model)   # attention is order-blind; this restores order
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        return self.head(self.norm(self.blocks(x)))

    @torch.no_grad()
    def write(self, idx, n=200, temperature=0.8, top_k=8):
        for _ in range(n):
            logits = self(idx[:, -self.block_size:])[:, -1] / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx
'''),
        md("### Train it"),
        code('''
corpus = (
    "in a hole in the ground there lived a hobbit not a nasty dirty wet hole "
    "filled with the ends of worms and an oozy smell nor yet a dry bare sandy "
    "hole with nothing in it to sit down on or to eat it was a hobbit hole and "
    "that means comfort "
) * 40

chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in corpus], device=device)
V = len(chars)

BS, BLOCK = 48, 48
def batch():
    ix = torch.randint(0, len(data) - BLOCK - 1, (BS,))
    return (torch.stack([data[i:i + BLOCK] for i in ix]),
            torch.stack([data[i + 1:i + BLOCK + 1] for i in ix]))

gpt = MiniGPT(V, block_size=BLOCK).to(device)
opt = torch.optim.AdamW(gpt.parameters(), lr=3e-3)
print("parameters:", sum(p.numel() for p in gpt.parameters()))

STEPS = 500    # 🔼 on a Colab GPU: 3000+ and a real corpus for dramatically better text
for step in range(STEPS):
    xs, ys = batch()
    loss = F.cross_entropy(gpt(xs).reshape(-1, V), ys.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 125 == 0:
        print(f"step {step:4d}: loss {loss.item():.3f}")
'''),
        code('''
seed = torch.tensor([[stoi["i"]]], device=device)
print("".join(itos[i] for i in gpt.write(seed, n=250)[0].tolist()))
'''),
        md('''
🎉 **That is a GPT.** Token embeddings + positional embeddings + a stack of
attention/MLP blocks + a next-token head. GPT-4 differs in scale (thousands of times more
parameters), tokenizer (subwords, not chars), data (the internet), and engineering — but you
just built the architecture, and you understand every line of it.
'''),
        registry_cell('''
_register("warmup", 5,
    lambda s: s.strip().lower() in ("v", "value", "values"),
    "attention output is a weighted sum of the ___ vectors (one letter)")
_register("attn_fn", 20,
    lambda f: (lambda o: torch.is_tensor(o) and o.shape == (2, 5, 8)
               and torch.allclose(f(torch.ones(1, 3, 4), torch.ones(1, 3, 4), torch.arange(12.).reshape(1, 3, 4)),
                                   torch.arange(12.).reshape(1, 3, 4).mean(1, keepdim=True).expand(1, 3, 4), atol=1e-5))(
        f(torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8))),
    "def attn(q, k, v): w = softmax(q @ k.transpose(-2,-1) / sqrt(d_k), dim=-1); return w @ v   (no mask)")
_register("causal_check", 15,
    lambda w: torch.is_tensor(w) and w.shape[0] == w.shape[1] and torch.allclose(w.triu(1), torch.zeros_like(w), atol=1e-6)
              and torch.allclose(w.sum(-1), torch.ones(w.shape[0]), atol=1e-5),
    "build a causal attention weight matrix for T=7 from random scores; upper triangle must be 0, rows sum to 1")
_register("gpt_low", 15,
    lambda l: l < 0.9,
    "train the MiniGPT until loss < 0.9 (more steps / slightly bigger model); submit loss.item()")
'''),
        code('''
check("warmup", "V")
'''),
        boss_md([
            '`attn_fn` (20 XP) — implement plain (unmasked) scaled dot-product attention `attn(q, k, v)`; submit the function.',
            '`causal_check` (15 XP) — build a valid `7×7` causal attention weight matrix from random scores; submit it.',
            '`gpt_low` (15 XP) — push MiniGPT training loss under `0.9`; submit the loss value.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("09_attention_build_a_gpt.ipynb", cells)


# ---------------------------------------------------------------------------
# 10 — The Debugging Dojo
# ---------------------------------------------------------------------------
def _q10():
    cells = [
        header("10_debugging_dojo.ipynb", "Act II", "10", "🥋 The Debugging Dojo",
               "Seven broken training loops. Seven classic PyTorch bugs. Diagnose each symptom, fix the code, earn your belt.",
               prev="09_attention_build_a_gpt.ipynb", nxt="11_art_of_creation.ipynb"),
        setup_cell(),
        md('''
## Welcome to the Dojo 🥋

Everything below **runs without crashing** — that's what makes these bugs dangerous. The model
just silently underperforms. For each station:

1. Run the broken code. **Study the symptom.**
2. Form a diagnosis *before* touching anything.
3. Write your fix in the attempt cell and grade it with `check(...)`.
4. Only then read the sensei's solution at the bottom.

These seven bugs account for a shocking fraction of all real-world "my model won't train" posts.
'''),
        code('''
# Shared training arena: a simple blob-classification task
def make_blobs(n=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2., 0.], [2., 0.], [0., 2.5]])
    X = torch.cat([c + 0.7 * torch.randn(n // 3, 2, generator=g) for c in centers])
    y = torch.arange(3).repeat_interleave(n // 3)
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]

Xb, yb = make_blobs()
def fresh_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3))

def acc(model, X=Xb, y=yb):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()
'''),
        md("## 🥋 Station 1 — The loss that refuses to settle"),
        code('''
model = fresh_model()
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for step in range(200):
    loss = F.cross_entropy(model(Xb), yb)
    loss.backward()
    opt.step()                    # 🐛 something is missing from this loop...
    if step % 40 == 0:
        print(f"step {step:3d}: loss {loss.item():8.3f}")
print("accuracy:", f"{acc(model)*100:.0f}%", " <- symptom: loss bounces/explodes instead of settling")
'''),
        md("**Diagnosis:** gradients *accumulate* across steps (Act I, Rule 1). Every step applies the sum of ALL past gradients. Fix it in the cell below."),
        code('''
# ⚔️ your fix — rewrite the loop correctly, then: check("station1", model)
'''),
        md("## 🥋 Station 2 — The loss that can't count"),
        code('''
model = fresh_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(300):
    opt.zero_grad()
    loss = F.mse_loss(model(Xb), yb.float().unsqueeze(1))   # 🐛 MSE... on class labels?!
    loss.backward(); opt.step()
print("accuracy:", f"{acc(model)*100:.0f}%", " <- symptom: loss falls but accuracy is garbage")
'''),
        md("**Diagnosis:** class indices (0,1,2) are *categories*, not quantities — regressing onto them treats class 2 as 'twice class 1'. Classification wants `cross_entropy` on logits."),
        code('''
# ⚔️ your fix here, then: check("station2", model)
'''),
        md("## 🥋 Station 3 — The rocket ship"),
        code('''
model = fresh_model()
opt = torch.optim.SGD(model.parameters(), lr=25.0)   # 🐛 lr=25?!
for step in range(100):
    opt.zero_grad()
    loss = F.cross_entropy(model(Xb), yb)
    loss.backward(); opt.step()
    if step % 25 == 0:
        print(f"step {step:3d}: loss {loss.item():.2e}")
print("symptom: loss shoots to infinity / NaN — every step overshoots the valley")
'''),
        code('''
# ⚔️ your fix here, then: check("station3", model)
'''),
        md("## 🥋 Station 4 — The silent broadcast"),
        code('''
# A regression task this time: predict y = 3x from noisy data
xr = torch.linspace(-2, 2, 200).unsqueeze(1)
yr = 3 * xr.squeeze() + 0.1 * torch.randn(200)        # 🐛 shape (200,) ... predictions are (200,1)

reg = nn.Linear(1, 1)
opt = torch.optim.Adam(reg.parameters(), lr=0.05)
for step in range(200):
    opt.zero_grad()
    loss = F.mse_loss(reg(xr).squeeze(-1) if False else reg(xr), yr)   # (200,1) vs (200,) -> (200,200)!!
    loss.backward(); opt.step()
print(f"learned weight: {reg.weight.item():.3f}  (should be ~3.0)")
print("symptom: loss 'trains' but the learned weight is wrong — broadcasting built a 200x200 error matrix")
'''),
        md("**Diagnosis:** `(200,1) - (200,)` broadcasts to `(200,200)` — MSE against every *pair*. Newer PyTorch warns about this; older versions are silent. Make the shapes match."),
        code('''
# ⚔️ your fix here — retrain reg so its weight is ~3, then: check("station4", reg)
'''),
        md("## 🥋 Station 5 — The frozen layer"),
        code('''
class TwoStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Linear(2, 16)
        self.stage2 = nn.Linear(16, 3)
    def forward(self, x):
        h = torch.relu(self.stage1(x))
        h = torch.tensor(h.detach().numpy())     # 🐛 out of the graph, then back in
        return self.stage2(h)

model = TwoStage()
before = model.stage1.weight.clone()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(200):
    opt.zero_grad()
    F.cross_entropy(model(Xb), yb).backward()
    opt.step()
print("stage1 weights changed:", not torch.allclose(before, model.stage1.weight))
print("accuracy:", f"{acc(model)*100:.0f}%", " <- symptom: trains 'okay' but stage1 never learns — gradient is severed")
'''),
        code('''
# ⚔️ your fix — a TwoStage whose stage1 actually learns, then: check("station5", model)
'''),
        md("## 🥋 Station 6 — The gambling evaluator"),
        code('''
drop_model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 3))
opt = torch.optim.Adam(drop_model.parameters(), lr=1e-2)
for step in range(300):
    opt.zero_grad()
    F.cross_entropy(drop_model(Xb), yb).backward()
    opt.step()

# 🐛 evaluating without model.eval() — dropout is still firing!
with torch.no_grad():
    a1 = (drop_model(Xb).argmax(1) == yb).float().mean().item()
    a2 = (drop_model(Xb).argmax(1) == yb).float().mean().item()
print(f"eval #1: {a1*100:.1f}%   eval #2: {a2*100:.1f}%   <- symptom: same data, different answers?!")
'''),
        code('''
# ⚔️ your fix — evaluate properly and submit the (deterministic) accuracy: check("station6", accuracy_value)
'''),
        md("## 🥋 Station 7 — The unshuffled deck"),
        code('''
from torch.utils.data import TensorDataset, DataLoader

# Data arrives SORTED by class (very common in real datasets!)
order = torch.argsort(yb)
Xs, ys = Xb[order], yb[order]

model = fresh_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loader = DataLoader(TensorDataset(Xs, ys), batch_size=32, shuffle=False)   # 🐛 shuffle=False
for epoch in range(3):
    for xb2, yb2 in loader:
        opt.zero_grad()
        F.cross_entropy(model(xb2), yb2).backward()
        opt.step()
print("accuracy:", f"{acc(model)*100:.0f}%",
      " <- symptom: each epoch ends having seen ONLY class-2 batches last; the model keeps 'forgetting'")
'''),
        code('''
# ⚔️ your fix here, then: check("station7", model)
'''),
        registry_cell('''
_TH = 0.90
_register("station1", 15, lambda m: acc(m) >= _TH, "add opt.zero_grad() at the top of the loop")
_register("station2", 15, lambda m: acc(m) >= _TH, "use F.cross_entropy(model(Xb), yb) — logits + integer labels")
_register("station3", 15, lambda m: acc(m) >= _TH, "a sane learning rate: 0.01–0.1 for SGD here")
_register("station4", 15, lambda r: abs(r.weight.item() - 3.0) < 0.2, "match shapes: compare (200,1) with (200,1), or squeeze to (200,)")
_register("station5", 15, lambda m: acc(m) >= _TH and isinstance(m, nn.Module), "delete the numpy round-trip — keep h as the tensor it was")
_register("station6", 15, lambda a: isinstance(a, float) and a >= _TH, "model.eval() before inference (and torch.no_grad()); submit the accuracy float")
_register("station7", 15, lambda m: acc(m) >= _TH, "shuffle=True in the DataLoader")
'''),
        md('''
---
## 🎓 Sensei's solutions

Scroll no further until you've earned your attempts. Each solution below runs and passes its check.
'''),
        code('''
# Station 1 — zero_grad
model = fresh_model()
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for step in range(200):
    opt.zero_grad()
    F.cross_entropy(model(Xb), yb).backward()
    opt.step()
check("station1", model)
'''),
        code('''
# Station 2 — right loss for the job
model = fresh_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(300):
    opt.zero_grad()
    F.cross_entropy(model(Xb), yb).backward()
    opt.step()
check("station2", model)
'''),
        code('''
# Station 3 — sane learning rate
model = fresh_model()
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for step in range(300):
    opt.zero_grad()
    F.cross_entropy(model(Xb), yb).backward()
    opt.step()
check("station3", model)
'''),
        code('''
# Station 4 — matching shapes
reg = nn.Linear(1, 1)
opt = torch.optim.Adam(reg.parameters(), lr=0.05)
yr_col = yr.unsqueeze(1)                      # (200,) -> (200,1)
for step in range(300):
    opt.zero_grad()
    F.mse_loss(reg(xr), yr_col).backward()
    opt.step()
check("station4", reg)
'''),
        code('''
# Station 5 — never leave the graph
class TwoStageFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Linear(2, 16)
        self.stage2 = nn.Linear(16, 3)
    def forward(self, x):
        return self.stage2(torch.relu(self.stage1(x)))

model = TwoStageFixed()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(300):
    opt.zero_grad()
    F.cross_entropy(model(Xb), yb).backward()
    opt.step()
check("station5", model)
'''),
        code('''
# Station 6 — eval mode for evaluation
drop_model.eval()
with torch.no_grad():
    a = (drop_model(Xb).argmax(1) == yb).float().mean().item()
drop_model.train()
check("station6", a)
'''),
        code('''
# Station 7 — shuffle your data
model = fresh_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loader = DataLoader(TensorDataset(Xs, ys), batch_size=32, shuffle=True)
for epoch in range(4):
    for xb2, yb2 in loader:
        opt.zero_grad()
        F.cross_entropy(model(xb2), yb2).backward()
        opt.step()
check("station7", model)

xp_report()
'''),
        md('''
---
## 🥋 Black belt earned

Commit the seven to memory — as *symptoms*, not just rules:

| Symptom | First suspect |
|---------|--------------|
| Loss bounces or climbs steadily | missing `zero_grad()` |
| Loss falls, accuracy is garbage | wrong loss for the task |
| Loss → `inf`/`NaN` in a few steps | learning rate too high |
| "Trains" but learns wrong values | silent broadcasting in the loss |
| One part of the model never improves | graph severed (`.detach()`, `.numpy()`, `.item()`) |
| Same input, different eval results | forgot `model.eval()` (dropout/batchnorm) |
| Loss cycles weirdly per epoch | unshuffled, class-sorted data |

**End of Act II.** Act III: creation (generative models), action (RL), and speed (deployment).
'''),
    ]
    write_notebook("10_debugging_dojo.ipynb", cells)
