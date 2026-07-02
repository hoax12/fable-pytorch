"""Act I — Forge the Engine (quests 01–05)."""
from nb_utils import (header, setup_cell, registry_cell, md, code, boss_md,
                      write_notebook)


# A compact autograd engine, forged in quest 02 and re-used in quest 04.
VALUE_CLASS = '''
class Value:
    """A scalar that remembers how it was made — and can compute gradients."""
    def __init__(self, data, _parents=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = _parents
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad         # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad  # d tanh/dx = 1 - tanh^2
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort: visit children before parents, then run the
        # chain rule backwards through the whole graph.
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # conveniences so expressions read naturally
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __repr__(self): return f"Value({self.data:.4f}, grad={self.grad:.4f})"
'''


def build():
    _q01()
    _q02()
    _q03()
    _q04()
    _q05()


# ---------------------------------------------------------------------------
# 01 — The Idea of Learning (no torch!)
# ---------------------------------------------------------------------------
def _q01():
    cells = [
        header("01_the_idea_of_learning.ipynb", "Act I", "01", "The Idea of Learning",
               "Before any framework: what IS learning? Knobs, loss, and rolling downhill — in pure Python.",
               nxt="02_build_your_own_autograd.ipynb"),
        setup_cell(torch_needed=False),
        md('''
## 🎮 A game: guess the machine

I'm thinking of a machine that turns numbers into numbers: `y = w * x`. You don't know `w`.
You only get to see examples. Your job — and the job of *every* neural network ever trained —
is to find the knob setting `w` that best explains the data.
'''),
        code('''
SECRET_W = 3.7   # 🤫 pretend you can't see this

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [SECRET_W * x for x in xs]
print("examples the machine produced:", list(zip(xs, ys)))
'''),
        md('''
### Step 1 — measure how wrong a guess is: the **loss**

Pick any guess for `w`. The **loss** is a single number scoring how badly it explains the data.
We'll use mean squared error: average of `(prediction − truth)²`.
'''),
        code('''
def loss(w):
    return sum((w * x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

for guess in [0.0, 2.0, 3.7, 6.0]:
    print(f"w={guess:>4}: loss = {loss(guess):8.3f}")
'''),
        code('''
ws = [i / 10 for i in range(-20, 100)]
plt.plot(ws, [loss(w) for w in ws])
plt.axvline(3.7, ls="--", c="green", label="secret w")
plt.xlabel("guess for w"); plt.ylabel("loss"); plt.legend()
plt.title("The loss is a valley. Learning = finding the bottom."); plt.show()
'''),
        md('''
### Step 2 — which way is downhill? the **gradient**

We could try every `w`... but a real network has *millions* of knobs. Instead, ask the slope:
nudge `w` a tiny bit and see how the loss reacts. That's a **finite-difference derivative**:

\\[ \\frac{dL}{dw} \\approx \\frac{L(w+h) - L(w-h)}{2h} \\]

Positive slope → downhill is to the *left*. Negative slope → downhill is to the *right*.
'''),
        code('''
def slope(f, w, h=1e-5):
    return (f(w + h) - f(w - h)) / (2 * h)

print("slope at w=1.0:", slope(loss, 1.0), " -> negative, so move RIGHT (increase w)")
print("slope at w=6.0:", slope(loss, 6.0), " -> positive, so move LEFT (decrease w)")
print("slope at w=3.7:", slope(loss, 3.7), " -> ~0, we'd be at the bottom")
'''),
        md('''
### Step 3 — roll downhill: **gradient descent**

Repeat: measure the slope, take a small step *against* it. The step size is the
**learning rate** — the single most important dial in deep learning.
'''),
        code('''
w = 0.0                      # start with a terrible guess
lr = 0.01                    # learning rate
history = [w]
for step in range(40):
    w = w - lr * slope(loss, w)
    history.append(w)

print(f"final guess: w = {w:.4f}   (secret was {SECRET_W})")
plt.plot(history, marker=".")
plt.axhline(SECRET_W, ls="--", c="green")
plt.xlabel("step"); plt.ylabel("w"); plt.title("Rolling downhill to the answer"); plt.show()
'''),
        md('''
### Step 4 — more knobs, same trick

Real models have many knobs. Here's `y = w*x + b` — two knobs. The recipe doesn't change:
compute the slope **for each knob**, step each one downhill. (The collection of all slopes is
called *the gradient*.)
'''),
        code('''
SECRET = (2.5, -1.0)          # secret w, b
data = [(x, SECRET[0] * x + SECRET[1]) for x in xs]

def loss2(w, b):
    return sum((w * x + b - y) ** 2 for x, y in data) / len(data)

w, b = 0.0, 0.0
for step in range(2000):
    dw = slope(lambda v: loss2(v, b), w)
    db = slope(lambda v: loss2(w, v), b)
    w, b = w - 0.01 * dw, b - 0.01 * db

print(f"learned: w={w:.3f}, b={b:.3f}   (secret {SECRET})")
'''),
        md('''
## 🧭 The universal recipe

Everything in this course — CNNs, GPTs, GANs — is this exact loop with fancier machines:

```
for each step:
    prediction = machine(inputs)                 # forward
    loss       = how_wrong(prediction, truth)    # measure
    gradient   = slope of loss w.r.t. EVERY knob # backward
    knobs     -= learning_rate * gradient        # descend
```

One problem: finite differences need **2 loss evaluations per knob per step**. GPT-4-class
models have ~10¹² knobs. We need something smarter than nudging — and in the next quest,
**you will build it**: an engine that computes *all* the slopes in one backward sweep.
'''),
        registry_cell('''
_register("warmup", 5,
    lambda ans: abs(ans - 42) < 1e-9,
    "the answer to everything")
_register("grad_fn", 15,
    lambda f: abs(f(lambda w: (w - 2) ** 2, 5.0) - 6.0) < 1e-3,
    "return (f(w+h) - f(w-h)) / (2*h) with a small h like 1e-5")
_register("descend", 15,
    lambda w: abs(w - 4.0) < 0.05,
    "run gradient descent on g(w) = (w-4)**2, ~100 steps with lr=0.1, starting anywhere")
_register("lr_quiz", 10,
    lambda s: s.strip().lower() == "diverge",
    "one word: what does the loss do if the learning rate is far too big — 'converge', 'diverge', or 'freeze'?")
'''),
        code('''
# warm-up: the grader in action. Run me!
check("warmup", 42)
'''),
        boss_md([
            '`grad_fn` (15 XP) — write `my_slope(f, w)` that estimates df/dw by finite differences, then `check("grad_fn", my_slope)`.',
            '`descend` (15 XP) — minimize `g(w) = (w - 4)**2` with your own descent loop; `check("descend", final_w)`.',
            '`lr_quiz` (10 XP) — `check("lr_quiz", "your one-word answer")`.',
        ]),
        code('''
# ⚔️ your attempts here...
# def my_slope(f, w, h=1e-5): ...
# check("grad_fn", my_slope)

# xp_report()
'''),
    ]
    write_notebook("01_the_idea_of_learning.ipynb", cells)


# ---------------------------------------------------------------------------
# 02 — Build Your Own Autograd
# ---------------------------------------------------------------------------
def _q02():
    cells = [
        header("02_build_your_own_autograd.ipynb", "Act I", "02", "Build Your Own Autograd",
               "Forge a ~60-line engine that computes every gradient in one sweep — then train a real neural net with it.",
               prev="01_the_idea_of_learning.ipynb", nxt="03_tensors_the_real_metal.ipynb"),
        setup_cell(torch_needed=False),
        md('''
## The key insight

Any computation is a **graph** of tiny operations (`+`, `*`, `tanh`...), and each tiny op has a
derivative we know from high-school calculus. The **chain rule** lets us multiply those local
derivatives backwards through the graph to get the slope of the final loss with respect to
*every* input — in **one** backward pass, no matter how many knobs.

That's it. That's the secret inside `loss.backward()`. Let's build it.

### The plan
A `Value` wraps a number and remembers:
- `data` — the number,
- `grad` — the slope of the final output w.r.t. this value (filled in later),
- `_parents` and `_backward` — who made it, and *how to push gradient back to them*.
'''),
        code(VALUE_CLASS),
        md('''
### Watch it think

Let's trace `f = (a * b + c).tanh()` with `a=2, b=-3, c=10` and ask for all three gradients at once.
'''),
        code('''
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
f = (a * b + c).tanh()
f.backward()

print("f     =", f.data)
print("df/da =", a.grad, "  (analytic: b * (1-tanh(ab+c)^2))")
print("df/db =", b.grad)
print("df/dc =", c.grad)

# Verify against finite differences — the slow method from Quest 01:
def numeric(name, delta=1e-5):
    vals = {"a": 2.0, "b": -3.0, "c": 10.0}
    up, dn = dict(vals), dict(vals)
    up[name] += delta; dn[name] -= delta
    g = lambda v: math.tanh(v["a"] * v["b"] + v["c"])
    return (g(up) - g(dn)) / (2 * delta)

for name, node in [("a", a), ("b", b), ("c", c)]:
    print(f"  {name}: engine={node.grad:.6f}  numeric={numeric(name):.6f}  ✓")
'''),
        md('''
One `backward()` call, every gradient, and they match the finite-difference check. **You just
built the core of PyTorch.**

### Now the real test: train a neural network with YOUR engine

A neuron is `tanh(w·x + b)`. A layer is several neurons. An MLP is stacked layers. All built
from `+`, `*`, and `tanh` — exactly the ops your engine speaks.
'''),
        code('''
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):          # e.g. MLP([2, 4, 1])
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

random.seed(1)
net = MLP([2, 4, 1])
print("your hand-forged network has", len(net.parameters()), "knobs")
'''),
        md('''
### The XOR problem — impossible for a single neuron, easy for your MLP

XOR: output +1 when the two inputs differ, −1 when they match. It's the classic test that
killed single-layer networks in 1969 and was solved by backprop in 1986. Your engine *is* backprop.
'''),
        code('''
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [-1.0, 1.0, 1.0, -1.0]

losses = []
for step in range(300):
    # forward
    preds = [net(x) for x in X]
    loss = sum((p - y) * (p - y) for p, y in zip(preds, Y))

    # backward — YOUR engine at work
    for p in net.parameters():
        p.grad = 0.0            # remember Quest 01: fresh slopes each step
    loss.backward()

    # descend
    for p in net.parameters():
        p.data -= 0.1 * p.grad
    losses.append(loss.data)

print("final predictions:", [f"{net(x).data:+.2f}" for x in X], " targets:", Y)
plt.plot(losses); plt.title("Your engine, training a neural net on XOR")
plt.xlabel("step"); plt.ylabel("loss"); plt.show()
'''),
        md('''
🎉 **You trained a neural network with an autograd engine you wrote from scratch.**

Everything PyTorch adds from here is *engineering*, not new ideas:
- tensors instead of scalars (do a million of these at once),
- GPU kernels (do them fast),
- a library of ops, layers, and optimizers (don't rewrite `Neuron` every day).

Next quest: meet the industrial version of what you just forged.
'''),
        registry_cell('''
_register("sub_works", 10,
    lambda v: isinstance(v, Value) and abs(v.data - 1.5) < 1e-9,
    "compute Value(4.0) - Value(2.5) — __sub__ is already defined via __add__ and __neg__")
_register("chain", 15,
    lambda g: abs(g - 12.0) < 1e-6,
    "y = x*x*x at x=2 -> dy/dx = 3x^2 = 12. Build it from Value(2.0), call .backward(), submit x.grad")
_register("xor_master", 20,
    lambda final_loss: final_loss < 0.05,
    "keep training (more steps, or lr=0.2) until the XOR loss is below 0.05, submit loss.data")
'''),
        boss_md([
            '`sub_works` (10 XP) — `check("sub_works", Value(4.0) - Value(2.5))`.',
            '`chain` (15 XP) — build `y = x*x*x` with `x = Value(2.0)`, backprop, then `check("chain", x.grad)`.',
            '`xor_master` (20 XP) — push the XOR training loss under `0.05`, then `check("xor_master", loss.data)`.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("02_build_your_own_autograd.ipynb", cells)


# ---------------------------------------------------------------------------
# 03 — Tensors, The Real Metal
# ---------------------------------------------------------------------------
def _q03():
    cells = [
        header("03_tensors_the_real_metal.ipynb", "Act I", "03", "Tensors — The Real Metal",
               "Your engine used one number at a time. Tensors do millions at once — here's the toolkit.",
               prev="02_build_your_own_autograd.ipynb", nxt="04_autograd_unmasked.ipynb"),
        setup_cell(),
        md('''
## Why your `Value` engine can't scale

Training XOR took 4 examples × 17 knobs. MNIST needs 60,000 images × ~100k knobs. Looping over
Python scalars would take *days*. A **tensor** is an n-dimensional array whose operations run in
optimized C/CUDA — the same math, thousands of times faster. Watch:
'''),
        code('''
import time

# dot product of 1M numbers: pure Python vs tensor
a_list = [random.random() for _ in range(1_000_000)]
b_list = [random.random() for _ in range(1_000_000)]

t0 = time.time()
s = sum(x * y for x, y in zip(a_list, b_list))
py_ms = (time.time() - t0) * 1000

a_t, b_t = torch.tensor(a_list), torch.tensor(b_list)
t0 = time.time()
s_t = a_t @ b_t
torch_ms = (time.time() - t0) * 1000

print(f"pure Python: {py_ms:7.1f} ms")
print(f"tensor     : {torch_ms:7.2f} ms   ({py_ms / max(torch_ms, 1e-6):.0f}x faster, same answer: {abs(s - s_t.item()) < 1e-3})")
'''),
        md("### Creating tensors & the big three properties: `shape`, `dtype`, `device`"),
        code('''
t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("shape:", t.shape, "| dtype:", t.dtype, "| device:", t.device)

print("\\nzeros:", torch.zeros(2, 3).shape)
print("random:", torch.randn(2, 3).shape)          # normal distribution
print("range:", torch.arange(0, 10, 2))
print("evenly spaced:", torch.linspace(0, 1, 5))
print("int -> float:", torch.arange(3).float().dtype)
# .to(device) moves tensors to the GPU when you have one (Colab!)
print("on device:", torch.randn(2).to(device).device)
'''),
        md("### Indexing & slicing — an image is just a tensor"),
        code('''
# Build a tiny "image" and manipulate it with pure indexing
img = torch.zeros(8, 8)
img[2:6, 2:6] = 1.0                     # a bright square

fig, ax = plt.subplots(1, 4, figsize=(10, 2.5))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("original")
ax[1].imshow(img.flip(1), cmap="gray"); ax[1].set_title("h-flip")
ax[2].imshow(img[::2, ::2], cmap="gray"); ax[2].set_title("2x downsample")
ax[3].imshow(img.T.roll(2, dims=0), cmap="gray"); ax[3].set_title("transpose+roll")
for a in ax: a.axis("off")
plt.show()

print("row 3:", img[3])
print("bright pixels:", (img > 0.5).sum().item())
'''),
        md('''
### Broadcasting — the superpower

Operating on mismatched shapes without loops. Rule: align shapes **from the right**; each pair
of dims must be equal, or one of them must be `1` (it stretches).

`(8,1) + (1,8) → (8,8)`  ·  `(N,3) - (3,) → (N,3)`
'''),
        code('''
# A radial gradient image in ONE line of broadcasting — no loops
n = 64
coord = torch.arange(n).float()
dist = ((coord[:, None] - n/2) ** 2 + (coord[None, :] - n/2) ** 2).sqrt()  # (64,1) & (1,64) -> (64,64)
plt.imshow(dist, cmap="magma"); plt.title("distance from center — pure broadcasting"); plt.colorbar(); plt.show()

# The most common real-world use: normalize features per column
data = torch.randn(200, 3) * torch.tensor([1., 10., 100.]) + torch.tensor([0., 5., -50.])
normed = (data - data.mean(dim=0)) / data.std(dim=0)     # (200,3) - (3,) broadcasts
print("means ~0:", normed.mean(0).round(decimals=4), " stds ~1:", normed.std(0).round(decimals=4))
'''),
        md("### Reshaping & matmul — the two ops you'll use every single day"),
        code('''
x = torch.arange(24.)
print("as (2,3,4):", x.reshape(2, 3, 4).shape)
print("flatten:", x.reshape(2, 3, 4).reshape(2, -1).shape, " (-1 = 'figure it out')")

batch = torch.randn(32, 1, 20, 20)                    # 32 glyph images
print("flatten batch for a linear layer:", batch.flatten(1).shape)   # keep dim 0

# Matmul IS the neural network: (batch, features) @ (features, out) -> (batch, out)
W = torch.randn(400, 10)
out = batch.flatten(1) @ W
print("layer output:", out.shape)

# Reductions
m = torch.tensor([[1., 5., 3.], [8., 2., 9.]])
print("\\nsum all:", m.sum().item(), "| per-column max:", m.max(dim=0).values, "| argmax per row:", m.argmax(dim=1))
'''),
        md('''
> ⚠️ **`view` vs `reshape`**: `view` requires contiguous memory; `reshape` always works. Use
> `reshape` until you have a reason not to. And **`permute` vs `reshape` are NOT the same** —
> `permute` reorders axes (moves data around), `reshape` just re-chops the same order.
'''),
        registry_cell('''
_register("warmup", 5,
    lambda t: torch.is_tensor(t) and t.shape == (3, 4) and (t == 7).all(),
    "torch.full((3, 4), 7.0) — or ones * 7")
_register("bullseye", 15,
    lambda img: torch.is_tensor(img) and img.shape == (32, 32)
                and img[16, 16] == 1 and img[0, 0] == 0
                and abs(img.float().mean().item() - ((((torch.arange(32.)[:,None]-16)**2 + (torch.arange(32.)[None,:]-16)**2).sqrt() <= 8).float().mean().item())) < 1e-6,
    "distance-from-center <= 8 -> 1.0 else 0.0, on a 32x32 grid, centered at (16,16). Broadcasting, no loops!")
_register("normalize", 15,
    lambda f: (lambda z: torch.allclose(z.mean(0), torch.zeros(4), atol=1e-5) and torch.allclose(z.std(0), torch.ones(4), atol=1e-4))(f(torch.randn(100, 4) * 7 + 3)),
    "def normalize(x): return (x - x.mean(0)) / x.std(0)")
_register("shape_oracle", 10,
    lambda s: tuple(s) == (16, 8, 5),
    "A is (16, 8, 32), B is (32, 5). What is (A @ B).shape? Submit a tuple.")
'''),
        code('''
check("warmup", torch.full((3, 4), 7.0))
'''),
        boss_md([
            '`bullseye` (15 XP) — a 32×32 tensor: 1.0 where distance from center (16,16) ≤ 8, else 0.0. **No loops.**',
            '`normalize` (15 XP) — write `normalize(x)` that standardizes each column; submit the *function*.',
            '`shape_oracle` (10 XP) — predict `(A @ B).shape` for `A:(16,8,32)`, `B:(32,5)` — submit a tuple *without running it*.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("03_tensors_the_real_metal.ipynb", cells)


# ---------------------------------------------------------------------------
# 04 — Autograd Unmasked
# ---------------------------------------------------------------------------
def _q04():
    cells = [
        header("04_autograd_unmasked.ipynb", "Act I", "04", "Autograd Unmasked",
               "PyTorch's engine, side-by-side with the one YOU forged. Same machine, industrial scale.",
               prev="03_tensors_the_real_metal.ipynb", nxt="05_your_first_real_network.ipynb"),
        setup_cell(),
        md('''
## The reveal

In Quest 02 you built `Value`: numbers that remember their history and backpropagate gradients.
PyTorch tensors do **exactly this** when you set `requires_grad=True`. Let's prove it — same
expression, your engine vs theirs.
'''),
        code(VALUE_CLASS),
        code('''
# ---- YOUR engine ----------------------------------------------------------
a1, b1, c1 = Value(2.0), Value(-3.0), Value(10.0)
f1 = (a1 * b1 + c1).tanh()
f1.backward()

# ---- PyTorch --------------------------------------------------------------
a2 = torch.tensor(2.0, requires_grad=True)
b2 = torch.tensor(-3.0, requires_grad=True)
c2 = torch.tensor(10.0, requires_grad=True)
f2 = (a2 * b2 + c2).tanh()
f2.backward()

print(f"{'':10s}{'your engine':>14s}{'pytorch':>12s}")
for name, mine, theirs in [("df/da", a1.grad, a2.grad), ("df/db", b1.grad, b2.grad), ("df/dc", c1.grad, c2.grad)]:
    print(f"{name:10s}{mine:14.6f}{theirs.item():12.6f}")
print("\\nIdentical. You already understand PyTorch's core.")
'''),
        md('''
### Peeking at the graph

Your `Value` stored `_parents` and `_op`. PyTorch stores the same thing in `grad_fn` — every
tensor knows the operation that created it.
'''),
        code('''
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = y + 5
w = z.sin()

node = w.grad_fn
print("walking the graph backwards from w:")
while node is not None:
    print("  ", type(node).__name__)
    node = node.next_functions[0][0] if node.next_functions else None
'''),
        md('''
### The three rules of living with autograd

**Rule 1 — gradients accumulate.** `backward()` *adds* into `.grad` (your engine did too:
`self.grad += ...`). Zero them between steps or they pile up.
'''),
        code('''
x = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"after backward #{i+1}: x.grad = {x.grad.item()}  (true dy/dx is 4 — it's stacking!)")
x.grad.zero_()
print("after zero_():", x.grad.item())
'''),
        md('''
**Rule 2 — only leaf tensors keep `.grad`.** A tensor you *created* with `requires_grad=True`
is a leaf. A tensor *computed from* others is not — its gradient flows through but isn't stored.
'''),
        code('''
w = torch.randn(3, requires_grad=True)      # leaf ✓
scaled = w * 2                               # NOT a leaf — it was computed
loss = (scaled ** 2).sum()
loss.backward()
print("w.grad:", w.grad)
print("scaled.grad:", scaled.grad, " <- None (with a warning if you try in some versions)")
print("\\n⚠️ classic bug: `w = torch.randn(3, requires_grad=True) * 0.1` makes w a NON-leaf!")
print("   fix: w = (torch.randn(3) * 0.1).requires_grad_(True)")
'''),
        md('''
**Rule 3 — turn tracking off when you don't need it.** Inference and weight updates shouldn't
build graphs: `torch.no_grad()` for blocks, `.detach()` for single tensors. Saves memory & time.
'''),
        code('''
w = torch.tensor(5.0, requires_grad=True)
with torch.no_grad():
    pred = w * 3                    # no graph built here
print("inside no_grad:", pred.requires_grad)

frozen = (w * 3).detach()           # snapshot, cut from the graph
print("detached:", frozen.requires_grad)
'''),
        md("### Full circle: gradient descent, torch edition"),
        code('''
# Minimize the Rosenbrock banana function — a classic torture test for optimizers
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

x = torch.tensor(-1.5, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
path = []
for step in range(4000):
    loss = rosenbrock(x, y)
    loss.backward()
    with torch.no_grad():
        x -= 1e-3 * x.grad
        y -= 1e-3 * y.grad
        x.grad.zero_(); y.grad.zero_()
    path.append((x.item(), y.item()))

xs = torch.linspace(-2, 2, 200); ys = torch.linspace(-1, 3, 200)
XX, YY = torch.meshgrid(xs, ys, indexing="xy")
plt.contourf(XX, YY, rosenbrock(XX, YY).log(), levels=30, cmap="viridis")
px, py = zip(*path[::50])
plt.plot(px, py, "r.-", ms=3, lw=1, label="descent path")
plt.plot(1, 1, "w*", ms=15, label="true minimum (1,1)")
plt.legend(); plt.title("Autograd navigating the banana valley"); plt.show()
print(f"reached ({x.item():.3f}, {y.item():.3f}) — target (1, 1)")
'''),
        registry_cell('''
_register("warmup", 5,
    lambda g: abs(g - 7.0) < 1e-5,
    "y = x**2 + 3x at x=2: dy/dx = 2x + 3 = 7. Build with requires_grad, backward, submit x.grad.item()")
_register("poly_grad", 15,
    lambda g: torch.is_tensor(g) and torch.allclose(g, torch.tensor([2., 4., 6.])),
    "w = torch.tensor([1.,2.,3.], requires_grad=True); loss = (w**2).sum(); backward; submit w.grad")
_register("leaf_fix", 15,
    lambda w: torch.is_tensor(w) and w.is_leaf and w.requires_grad and w.shape == (4,),
    "create a shape-(4,) leaf tensor scaled by 0.1 that still tracks gradients: (torch.randn(4)*0.1).requires_grad_(True)")
_register("valley", 15,
    lambda xy: abs(xy[0] - 1) < 0.15 and abs(xy[1] - 1) < 0.15,
    "run the Rosenbrock descent longer (or nudge lr) until (x,y) is within 0.15 of (1,1); submit (x.item(), y.item())")
'''),
        code('''
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
check("warmup", x.grad.item())
'''),
        boss_md([
            '`poly_grad` (15 XP) — gradient of `(w**2).sum()` at `w=[1,2,3]`; submit `w.grad`.',
            '`leaf_fix` (15 XP) — create a *scaled* leaf tensor that keeps `.grad` (the Rule-2 bug, fixed); submit it.',
            '`valley` (15 XP) — get the banana-valley descent within `0.15` of `(1, 1)`; submit `(x.item(), y.item())`.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("04_autograd_unmasked.ipynb", cells)


# ---------------------------------------------------------------------------
# 05 — Your First Real Network
# ---------------------------------------------------------------------------
def _q05():
    cells = [
        header("05_your_first_real_network.ipynb", "Act I", "05", "Your First Real Network",
               "nn.Module, optimizers, losses — the industrial version of the MLP you hand-forged.",
               prev="04_autograd_unmasked.ipynb", nxt="06_feeding_the_beast.ipynb"),
        setup_cell(),
        md('''
## From your `MLP` class to `nn.Module`

In Quest 02 you wrote `Neuron`, `Layer`, `MLP` and tracked parameters by hand. PyTorch's
`nn.Module` is that pattern, productionized: it auto-registers parameters, moves them between
devices, saves/loads them, and composes into arbitrarily deep trees of modules.

The mapping is exact:

| your forge | PyTorch |
|------------|---------|
| `Neuron` / `Layer` | `nn.Linear` |
| `.tanh()` in the neuron | separate activation (`nn.Tanh`, `nn.ReLU`) |
| `net.parameters()` list | `model.parameters()` generator |
| `p.data -= lr * p.grad` loop | `optimizer.step()` |
| `p.grad = 0` loop | `optimizer.zero_grad()` |
| `sum((p-y)**2)` | `nn.MSELoss`, `nn.CrossEntropyLoss`, ... |
'''),
        md("### The arena: two interlocking spirals (much nastier than XOR)"),
        code('''
def make_spirals(n=600, noise=0.12, turns=3):
    n2 = n // 2
    idx = torch.arange(n2).float()
    r = idx / n2 * 1.6
    th = idx / n2 * turns * torch.pi
    s0 = torch.stack([r * torch.sin(th), r * torch.cos(th)], dim=1)
    s1 = torch.stack([r * torch.sin(th + torch.pi), r * torch.cos(th + torch.pi)], dim=1)
    X = torch.cat([s0, s1]) + noise * torch.randn(n, 2)
    y = torch.cat([torch.zeros(n2), torch.ones(n2)]).long()
    perm = torch.randperm(n)
    return X[perm], y[perm]

X, y = make_spirals()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10)
plt.title("Two spirals — try drawing a straight line through THAT"); plt.axis("equal"); plt.show()
'''),
        md("### Define the model"),
        code('''
class SpiralNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),          # 2 logits: one score per class
        )
    def forward(self, x):
        return self.body(x)

model = SpiralNet().to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))
'''),
        md('''
### The five-line liturgy

You will write this loop hundreds of times in your life. Learn it as a rhythm:

```
zero → forward → loss → backward → step
```
'''),
        code('''
X, y = X.to(device), y.to(device)
criterion = nn.CrossEntropyLoss()                       # softmax + NLL in one, expects raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

losses = []
for epoch in range(600):
    optimizer.zero_grad()          # 1. zero   (Rule 1 of autograd!)
    logits = model(X)              # 2. forward
    loss = criterion(logits, y)    # 3. loss
    loss.backward()                # 4. backward
    optimizer.step()               # 5. step
    losses.append(loss.item())

acc = (model(X).argmax(1) == y).float().mean().item()
print(f"loss {losses[-1]:.4f} | accuracy {acc*100:.1f}%")
plt.plot(losses); plt.title("The liturgy, working"); plt.xlabel("epoch"); plt.show()
'''),
        code('''
# The decision boundary — compare with what a straight line could ever do
def plot_boundary(model, X, y):
    xs = torch.linspace(X[:, 0].min() - .3, X[:, 0].max() + .3, 250)
    ys = torch.linspace(X[:, 1].min() - .3, X[:, 1].max() + .3, 250)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([XX.flatten(), YY.flatten()], dim=1).to(device)
    with torch.no_grad():
        Z = model(grid).argmax(1).reshape(XX.shape).cpu()
    plt.contourf(XX, YY, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), cmap="coolwarm", s=8, edgecolors="k", linewidths=0.2)
    plt.axis("equal")

plot_boundary(model, X, y)
plt.title("A learned spiral boundary — this is why depth matters"); plt.show()
'''),
        md('''
### Optimizers: smarter ways to roll downhill

Plain SGD steps straight down the local slope. **Momentum** remembers velocity; **Adam** adapts
a per-knob learning rate. Same valley, different descent styles:
'''),
        code('''
results = {}
for name, make_opt in {
    "SGD":      lambda p: torch.optim.SGD(p, lr=0.05),
    "SGD+mom":  lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "Adam":     lambda p: torch.optim.Adam(p, lr=0.01),
}.items():
    torch.manual_seed(7)
    m = SpiralNet().to(device)
    opt = make_opt(m.parameters())
    hist = []
    for _ in range(300):
        opt.zero_grad()
        l = criterion(m(X), y)
        l.backward(); opt.step()
        hist.append(l.item())
    results[name] = hist

for name, hist in results.items():
    plt.plot(hist, label=name)
plt.legend(); plt.title("Same net, three optimizers"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
'''),
        md('''
### `train()` / `eval()` and saving your work

Two module modes: `model.train()` (dropout on, batchnorm updating) and `model.eval()`
(deterministic inference). Pair `eval()` with `no_grad()`. And save the `state_dict` — just the
weights, not the code.
'''),
        code('''
import os
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/spiralnet.pt")

revived = SpiralNet().to(device)
revived.load_state_dict(torch.load("checkpoints/spiralnet.pt", map_location=device))
revived.eval()
with torch.no_grad():
    same = torch.equal(revived(X).argmax(1), model(X).argmax(1))
print("revived model agrees with original:", same)
'''),
        registry_cell('''
_register("warmup", 5,
    lambda n: n == sum(p.numel() for p in nn.Linear(10, 5).parameters()),
    "nn.Linear(10, 5) has 10*5 weights + 5 biases = 55")
_register("liturgy", 15,
    lambda steps: [s.strip().lower() for s in steps] == ["zero", "forward", "loss", "backward", "step"],
    "submit the 5 steps of the training loop as a list of lowercase words")
_register("spiral_95", 20,
    lambda m: isinstance(m, nn.Module) and (m(X).argmax(1) == y).float().mean().item() >= 0.95,
    "train a model (any architecture) to >= 95% accuracy on the spirals X, y and submit the model")
_register("custom_block", 15,
    lambda m: isinstance(m, nn.Module) and m(torch.randn(3, 8)).shape == (3, 8)
              and any(p.numel() for p in m.parameters()),
    "an nn.Module whose forward returns x + linear(x).relu() — a residual block! in: (B,8), out: (B,8)")
'''),
        code('''
check("warmup", 55)
'''),
        boss_md([
            '`liturgy` (15 XP) — the five steps, in order, as a list of words.',
            '`spiral_95` (20 XP) — train any model to ≥95% on the spirals; submit the trained model.',
            '`custom_block` (15 XP) — write a **residual block** module: `forward(x) = x + relu(linear(x))` for 8-dim inputs; submit an instance.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
        md('''
---
## 🏁 End of Act I

Take stock of what you can now do: derive learning from scratch, **build an autograd engine**,
wield tensors, and train real networks with the five-line liturgy. Act II puts specialized
senses on your networks: eyes (convolutions), memory (recurrence), and attention (GPT).
'''),
    ]
    write_notebook("05_your_first_real_network.ipynb", cells)
