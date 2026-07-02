"""Epilogue quest 15 — Beyond the Forge (ONNX deployment)."""
from nb_utils import (header, setup_cell, registry_cell, md, code, boss_md,
                      write_notebook, GLYPH_DATA)


ONNX_ENSURE = '''
# onnx + onnxruntime: auto-install on Colab if missing
import sys, subprocess, importlib
for pkg in ["onnx", "onnxruntime"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import onnx
import onnxruntime as ort
import os, time, json
print(f"onnx {onnx.__version__} | onnxruntime {ort.__version__}")
'''


def build():
    cells = [
        header("15_beyond_the_forge_onnx.ipynb", "Epilogue", "15", "Beyond the Forge — ONNX",
               "Your model must now survive OUTSIDE the forge: no PyTorch, no Python class, just a portable artifact.",
               prev="14_final_boss.ipynb"),
        setup_cell(extra=ONNX_ENSURE),
        md('''
## The last door

Everything you've trained so far lives *inside* PyTorch — a Python object that needs your class
definition and the whole torch runtime to make a single prediction. Production usually can't
afford that: a C++ game server, a phone, a browser tab, a tiny container.

**ONNX** (Open Neural Network Exchange) is the passport out:

```
PyTorch ──torch.onnx.export──▶ model.onnx ──▶ ONNX Runtime · TensorRT · CoreML · the browser
```

**ONNX Runtime** executes `.onnx` files anywhere, usually *faster* than eager PyTorch on CPU —
and it needs only `onnxruntime` + `numpy` at serving time.

### The quest
1. 🏗️ Train a glyph classifier (your last one — savor it)
2. 📦 Export to ONNX with a **dynamic batch axis**
3. 🔍 Open the artifact and inspect the graph inside
4. ✅ Serve with ONNX Runtime; prove it matches PyTorch
5. ⏱️ Race them: eager vs ONNX Runtime
6. 🪶 Quantize the artifact to int8
7. 🚀 A torch-free oracle (`ship/glyph_oracle.py`)
'''),
        md("## 1 · 🏗️ Train the traveler"),
        code(GLYPH_DATA),
        code('''
from torch.utils.data import TensorDataset, DataLoader

X, y = make_glyphs(n_per_class=300)
train_loader = DataLoader(TensorDataset(X[:960], y[:960]), batch_size=64, shuffle=True)
X_test, y_test = X[960:], y[960:]

traveler = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 20 -> 10
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 10 -> 5
    nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 4),
).to(device)
opt = torch.optim.Adam(traveler.parameters(), lr=1.5e-3)
for epoch in range(4):
    traveler.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); F.cross_entropy(traveler(xb), yb).backward(); opt.step()

traveler.eval()
with torch.no_grad():
    acc = (traveler(X_test.to(device)).argmax(1).cpu() == y_test).float().mean()
print(f"traveler trained — test accuracy {acc*100:.1f}%")
'''),
        md('''
## 2 · 📦 Export

`torch.onnx.export` traces the model once with an example input, records the graph, and writes
it out. Two details matter:

- **`dynamic_shapes=({0: "batch"},)`** marks dim 0 as symbolic — without it, the artifact only
  accepts the exact batch size you exported with. (Older code uses `dynamic_axes=`; same idea.)
- **`external_data=False`** embeds the weights so you get ONE portable file instead of
  `glyphs.onnx` + a `glyphs.onnx.data` sidecar.
'''),
        code('''
os.makedirs("ship", exist_ok=True)
PATH = "ship/glyphs.onnx"

traveler.cpu().eval()
torch.onnx.export(
    traveler, (torch.randn(1, 1, 20, 20),), PATH,
    input_names=["glyph"], output_names=["logits"],
    dynamic_shapes=({0: "batch"},),      # dim 0 of the (only) input is symbolic
    opset_version=18,
    external_data=False,                 # one self-contained file
)
with open("ship/glyph_classes.json", "w") as f:
    json.dump(GLYPHS, f)
print(f"\\nexported {PATH} ({os.path.getsize(PATH) / 1e3:.1f} KB) + class names")
'''),
        md("## 3 · 🔍 What's inside the artifact?"),
        code('''
model_proto = onnx.load(PATH)
onnx.checker.check_model(model_proto)      # raises if malformed
print("graph valid ✅\\n")

def dims(t):
    return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]

print("input :", [(i.name, dims(i)) for i in model_proto.graph.input])
print("output:", [(o.name, dims(o)) for o in model_proto.graph.output])
print("ops   :", sorted({n.op_type for n in model_proto.graph.node}))
'''),
        md('''
The batch dimension reads `"batch"` — a *symbol*, not a number. And your `nn.Sequential` has
become a flat list of standard ops (`Conv`, `Relu`, `MaxPool`, `Gemm`...) any runtime can execute.
No Python. No class. No torch.
'''),
        md("## 4 · ✅ Serve it — and prove nothing was lost in translation"),
        code('''
sess = ort.InferenceSession(PATH, providers=["CPUExecutionProvider"])

x_np = X_test.numpy()
with torch.no_grad():
    torch_logits = traveler(X_test).numpy()
ort_logits = sess.run(["logits"], {"glyph": x_np})[0]

diff = np.abs(torch_logits - ort_logits).max()
print(f"max |pytorch − onnxruntime| = {diff:.2e}  -> match: {diff < 1e-4} ✅")

# dynamic batch in action: same artifact, any batch size
for b in (1, 7, 64):
    out = sess.run(["logits"], {"glyph": np.random.rand(b, 1, 20, 20).astype(np.float32)})[0]
    print(f"batch {b:2d} -> logits {out.shape}")
'''),
        md("## 5 · ⏱️ The race — eager PyTorch vs ONNX Runtime (CPU)"),
        code('''
big = np.random.rand(128, 1, 20, 20).astype(np.float32)
big_t = torch.from_numpy(big)

def clock(fn, reps=60):
    fn()
    t0 = time.time()
    for _ in range(reps):
        fn()
    return (time.time() - t0) / reps * 1000

t_torch = clock(lambda: traveler(big_t))
t_ort = clock(lambda: sess.run(["logits"], {"glyph": big}))
print(f"PyTorch eager: {t_torch:6.2f} ms/batch")
print(f"ONNX Runtime : {t_ort:6.2f} ms/batch   ({t_torch / t_ort:.2f}x)")
'''),
        md('''
## 6 · 🪶 Quantize the artifact

Dynamic int8 quantization converts weights to 8-bit integers, directly on the `.onnx` file —
no PyTorch involved. One real-world wrinkle: the exporter bakes cached activation shapes
(`value_info`) into the graph, and they confuse the quantizer's shape inference when the batch
axis is symbolic. We strip them first. (Welcome to deployment: half the job is little
compatibility moves like this.)
'''),
        code('''
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx(src, dst):
    """Strip stale value_info (breaks shape inference with dynamic axes), then quantize."""
    m = onnx.load(src)
    del m.graph.value_info[:]
    tmp = src + ".clean"
    onnx.save(m, tmp)
    quantize_dynamic(tmp, dst, weight_type=QuantType.QInt8)
    os.remove(tmp)

Q8 = "ship/glyphs_int8.onnx"
quantize_onnx(PATH, Q8)

kb, kb8 = os.path.getsize(PATH) / 1e3, os.path.getsize(Q8) / 1e3
print(f"fp32 {kb:.1f} KB -> int8 {kb8:.1f} KB   ({kb / kb8:.1f}x smaller)")

qsess = ort.InferenceSession(Q8, providers=["CPUExecutionProvider"])
q_acc = (qsess.run(["logits"], {"glyph": x_np})[0].argmax(1) == y_test.numpy()).mean()
print(f"int8 accuracy: {q_acc*100:.1f}%  (fp32 was {acc*100:.1f}%)")
'''),
        md('''
Roughly 4x smaller (float32 → int8), accuracy essentially untouched. For our toy model the KBs
don't matter — for a 7B-parameter LLM, this same idea is the difference between "fits on your
GPU" and "doesn't".
'''),
        md('''
## 7 · 🚀 The oracle — inference with zero torch

This is the whole point. The class below imports **only numpy + onnxruntime**. Drop it and the
two `ship/` files on any machine and you have a prediction service. A standalone CLI version
lives at [`ship/glyph_oracle.py`](../ship/glyph_oracle.py) — run it from `fable_folder/`:

```bash
python ship/glyph_oracle.py notebooks/ship/glyphs.onnx --classes notebooks/ship/glyph_classes.json --glyph ring
```
'''),
        code('''
class GlyphOracle:
    """Torch-free serving: numpy in, prediction out."""
    def __init__(self, model_path, classes_path):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        with open(classes_path) as f:
            self.classes = json.load(f)

    def __call__(self, image_20x20):
        x = np.asarray(image_20x20, dtype=np.float32).reshape(1, 1, 20, 20)
        logits = self.sess.run(None, {self.inp: x})[0][0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        k = int(p.argmax())
        return {"glyph": self.classes[k], "confidence": float(p[k])}

oracle = GlyphOracle("ship/glyphs.onnx", "ship/glyph_classes.json")
sample_idx = int((y_test == 1).nonzero()[0])       # grab a ring ◯ from the test set
print("oracle says:", oracle(X_test[sample_idx, 0].numpy()),
      "| truth:", GLYPHS[y_test[sample_idx]])
'''),
        registry_cell('''
_register("warmup", 5,
    lambda s: "torch" in s.lower() or "pytorch" in s.lower() or "python" in s.lower(),
    "what dependency does an .onnx artifact free your serving code from?")
_register("dyn_batch", 20,
    lambda s: (lambda n: s.run(None, {n: np.zeros((1, 1, 20, 20), np.float32)})[0].shape[0] == 1
               and s.run(None, {n: np.zeros((5, 1, 20, 20), np.float32)})[0].shape[0] == 5)(s.get_inputs()[0].name),
    "export ANY glyph model with a dynamic dim-0 (dynamic_shapes / dynamic_axes) and submit an ort.InferenceSession for it")
_register("faithful", 15,
    lambda d: float(d) < 1e-4,
    "submit the max abs difference between pytorch and onnxruntime logits on the test set")
_register("quant_shrunk", 15,
    lambda sizes: len(sizes) == 2 and sizes[1] < sizes[0],
    "submit (fp32_kb, int8_kb) — int8 must be smaller. Remember the lesson: use a big-enough model (the chunky one qualifies)")
'''),
        code('''
check("warmup", "torch")
'''),
        boss_md([
            '`dyn_batch` (20 XP) — export a model with a dynamic batch axis; submit the `InferenceSession`.',
            '`faithful` (15 XP) — submit the max PyTorch↔ORT logit difference as a float.',
            '`quant_shrunk` (15 XP) — submit `(fp32_kb, int8_kb)` after quantizing.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
        md('''
---
## 🌅 Epilogue complete

Your model walked out of the forge and works in a world that has never heard of PyTorch. That's
the full arc: **idea → engine → framework → mastery → artifact.**

Ideas for your own adventures: serve the oracle from FastAPI, run the `.onnx` in the browser
with ONNX Runtime Web, or export your Final Boss model and quantize it. ⚒️
'''),
    ]
    write_notebook("15_beyond_the_forge_onnx.ipynb", cells)
