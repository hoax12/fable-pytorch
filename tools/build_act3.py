"""Act III — Master the Arts (quests 11–14)."""
from nb_utils import (header, setup_cell, registry_cell, md, code, boss_md,
                      write_notebook, GLYPH_DATA)


def build():
    _q11()
    _q12()
    _q13()
    _q14()


# ---------------------------------------------------------------------------
# 11 — The Art of Creation
# ---------------------------------------------------------------------------
def _q11():
    cells = [
        header("11_art_of_creation.ipynb", "Act III", "11", "The Art of Creation",
               "Networks that don't just recognize — they invent. Autoencoders, VAEs, GANs, and diffusion.",
               prev="10_debugging_dojo.ipynb", nxt="12_art_of_action.ipynb"),
        setup_cell(),
        md('''
## Flipping the question

So far: *"given this input, what is it?"* Generative modeling asks the reverse:
*"what does the data **distribution** look like — and can I draw new samples from it?"*

Four ideas, in ascending order of magic. All run in seconds on CPU.

| Model | Core trick |
|-------|-----------|
| **Autoencoder** | squeeze data through a bottleneck; whatever survives is the essence |
| **VAE** | make the bottleneck a *distribution* → sampling becomes possible |
| **GAN** | a forger and a detective train against each other |
| **Diffusion** | learn to undo noise, then sculpt samples out of pure static |
'''),
        md("### 1 — Autoencoder: compress glyphs through a 2-D keyhole"),
        code(GLYPH_DATA),
        code('''
X, y = make_glyphs(n_per_class=250)
X_flat = X.flatten(1)                                   # (1000, 400)

class AutoEncoder(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(400, 128), nn.ReLU(), nn.Linear(128, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 400), nn.Sigmoid())
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

ae = AutoEncoder().to(device)
opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
Xd = X_flat.to(device)
for epoch in range(400):
    recon, z = ae(Xd)
    loss = F.mse_loss(recon, Xd)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"reconstruction MSE: {loss.item():.4f}   (400 pixels squeezed through 2 numbers!)")
'''),
        code('''
# Left: originals vs reconstructions. Right: THE LATENT SPACE — the 2 numbers, colored by class.
fig = plt.figure(figsize=(11, 3.5))
with torch.no_grad():
    recon, z = ae(Xd)
for i in range(6):
    axo = fig.add_subplot(2, 12, i + 1);      axo.imshow(X[i, 0], cmap="gray"); axo.axis("off")
    axr = fig.add_subplot(2, 12, i + 13);     axr.imshow(recon[i].reshape(20, 20).cpu(), cmap="gray"); axr.axis("off")
axz = fig.add_subplot(1, 2, 2)
sc = axz.scatter(z[:, 0].cpu(), z[:, 1].cpu(), c=y, cmap="tab10", s=8)
axz.legend(handles=sc.legend_elements()[0], labels=GLYPHS, fontsize=8)
axz.set_title("latent space: the AE separated the classes ON ITS OWN")
plt.suptitle("top: originals · bottom: reconstructions"); plt.show()
'''),
        md('''
Nobody told the autoencoder about classes — it discovered the glyph types just by learning to
compress. **Representation learning** in one picture.

### 2 — VAE: make the keyhole sample-able

An AE's latent space has *holes* — decode a random point and you get junk. A **VAE** encodes
each input as a *Gaussian* (`μ`, `σ`) and penalizes the latents for straying from `N(0, I)`.
Result: a smooth space you can sample from.
'''),
        code('''
class VAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(400, 128), nn.ReLU())
        self.mu = nn.Linear(128, latent)
        self.logvar = nn.Linear(128, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 400), nn.Sigmoid())
    def forward(self, x):
        h = self.body(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)    # the reparameterization trick
        return self.dec(z), mu, logvar

vae = VAE().to(device)
opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
for epoch in range(500):
    recon, mu, logvar = vae(Xd)
    rec = F.mse_loss(recon, Xd)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec + 0.005 * kl
    opt.zero_grad(); loss.backward(); opt.step()
print(f"recon {rec.item():.4f} | KL {kl.item():.3f}")

# Sample brand-new glyphs from pure noise!
with torch.no_grad():
    fresh = vae.dec(torch.randn(12, 2).to(device)).reshape(-1, 20, 20).cpu()
fig, ax = plt.subplots(1, 12, figsize=(12, 1.5))
for i in range(12):
    ax[i].imshow(fresh[i], cmap="gray"); ax[i].axis("off")
plt.suptitle("glyphs that never existed — decoded from random latent points"); plt.show()
'''),
        md('''
### 3 — GAN: the forger and the detective

Target: points arranged in a **heart** ♥. The **generator** G never sees the data — it only
gets feedback from the **discriminator** D, which is simultaneously learning to tell real from
fake. Two networks locked in an arms race.
'''),
        code('''
def sample_heart(n):
    t = torch.rand(n) * 2 * math.pi
    x = 16 * torch.sin(t) ** 3
    y = 13 * torch.cos(t) - 5 * torch.cos(2 * t) - 2 * torch.cos(3 * t) - torch.cos(4 * t)
    return (torch.stack([x, y], dim=1) / 16.0) + 0.03 * torch.randn(n, 2)

real = sample_heart(2000).to(device)

G = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
D = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1)).to(device)
optG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

snapshots = {}
for step in range(4000):
    # detective: real -> 1, forged -> 0
    idx = torch.randint(0, len(real), (256,))
    fake = G(torch.randn(256, 8, device=device)).detach()
    lossD = bce(D(real[idx]), torch.ones(256, 1, device=device)) + \\
            bce(D(fake), torch.zeros(256, 1, device=device))
    optD.zero_grad(); lossD.backward(); optD.step()

    # forger: make D say 1
    fake = G(torch.randn(256, 8, device=device))
    lossG = bce(D(fake), torch.ones(256, 1, device=device))
    optG.zero_grad(); lossG.backward(); optG.step()

    if step in (0, 500, 1500, 3999):
        with torch.no_grad():
            snapshots[step] = G(torch.randn(1500, 8, device=device)).cpu()

fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
for i, (s, pts) in enumerate(snapshots.items()):
    ax[i].scatter(real[:, 0].cpu(), real[:, 1].cpu(), s=3, alpha=0.15, c="gray")
    ax[i].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, c="crimson")
    ax[i].set_title(f"step {s}"); ax[i].set_xlim(-1.6, 1.6); ax[i].set_ylim(-1.6, 1.2); ax[i].axis("off")
plt.suptitle("the forger learns the heart, having NEVER seen it directly ♥"); plt.show()
'''),
        md('''
### 4 — Diffusion: sculpting from static

The idea behind Stable Diffusion, in miniature. **Forward**: gradually noise the data to pure
static over `T` steps. **Learn**: a network that predicts the added noise at any step.
**Generate**: start from static, subtract predicted noise step by step.
'''),
        code('''
T = 60
betas = torch.linspace(1e-4, 0.04, T)
alpha_bar = torch.cumprod(1 - betas, dim=0).to(device)

class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T).unsqueeze(-1)], dim=-1))

eps_net = NoisePredictor().to(device)
opt = torch.optim.Adam(eps_net.parameters(), lr=1e-3)
for step in range(4000):
    x0 = sample_heart(256).to(device)
    t = torch.randint(0, T, (256,), device=device)
    noise = torch.randn_like(x0)
    ab = alpha_bar[t].unsqueeze(-1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise      # jump straight to noise level t
    loss = F.mse_loss(eps_net(xt, t), noise)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"noise-prediction loss: {loss.item():.4f}")
'''),
        code('''
@torch.no_grad()
def sculpt(n=1500):
    x = torch.randn(n, 2, device=device)               # pure static
    frames = [x.cpu().clone()]
    for t in reversed(range(T)):
        eps = eps_net(x, torch.full((n,), t, device=device))
        a, ab = 1 - betas[t], alpha_bar[t]
        x = (x - betas[t].to(device) / (1 - ab).sqrt() * eps) / a.sqrt().to(device)
        if t > 0:
            x = x + betas[t].sqrt().to(device) * torch.randn_like(x)
        if t in (T - 1, T // 2, T // 5, 0):
            frames.append(x.cpu().clone())
    return frames

frames = sculpt()
fig, ax = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
for i, f in enumerate(frames):
    ax[i].scatter(f[:, 0], f[:, 1], s=3, alpha=0.5, c="purple")
    ax[i].set_xlim(-2, 2); ax[i].set_ylim(-2, 2); ax[i].axis("off")
    ax[i].set_title("static" if i == 0 else f"denoising…" if i < len(frames) - 1 else "♥")
plt.suptitle("diffusion: a heart sculpted out of pure noise"); plt.show()
'''),
        md('''
Swap the 2-D points for images and the MLP for a U-Net, and this *identical algorithm* is
Stable Diffusion. You now hold the core of modern generative AI.
'''),
        registry_cell('''
_register("warmup", 5,
    lambda s: "bottleneck" in s.lower() or "compress" in s.lower() or "latent" in s.lower() or "squeeze" in s.lower(),
    "one word for what forces an autoencoder to learn structure")
_register("tight_ae", 15,
    lambda mse: mse < 0.02,
    "improve the AE (bigger hidden layer, latent=4, more epochs) until recon MSE < 0.02; submit loss.item()")
_register("latent_walk", 15,
    lambda imgs: torch.is_tensor(imgs) and imgs.shape == (8, 400),
    "decode 8 points interpolated between two random latent vectors: z = a*(1-t) + b*t; submit vae.dec(zs)")
_register("gan_roles", 10,
    lambda s: s.strip().lower() in ("discriminator", "detective", "critic", "d"),
    "which network provides ALL of the generator's learning signal?")
'''),
        code('''
check("warmup", "bottleneck")
'''),
        boss_md([
            '`tight_ae` (15 XP) — get the autoencoder MSE under `0.02`; submit the loss value.',
            '`latent_walk` (15 XP) — interpolate 8 steps between two random latent points and decode them; submit the `(8, 400)` tensor.',
            '`gan_roles` (10 XP) — one word: where does the generator\'s learning signal come from?',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("11_art_of_creation.ipynb", cells)


# ---------------------------------------------------------------------------
# 12 — The Art of Action
# ---------------------------------------------------------------------------
def _q12():
    cells = [
        header("12_art_of_action.ipynb", "Act III", "12", "The Art of Action",
               "No labels, no targets — just consequences. Reinforcement learning on CartPole.",
               prev="11_art_of_creation.ipynb", nxt="13_art_of_speed.ipynb"),
        setup_cell(),
        md('''
## Learning from consequences

Everything so far had ground-truth answers. In **reinforcement learning** an *agent* acts in an
*environment* and receives only *rewards* — no one ever says which action was correct. The agent
must discover a **policy** (state → action) that maximizes total reward.

Our arena: **CartPole** — push a cart left/right to keep a pole balanced. Reward: +1 per tick
alive. If `gymnasium` is installed we use it; otherwise a built-in physics clone kicks in.
'''),
        code('''
class MiniCartPole:
    """Physics-faithful CartPole clone so the quest never needs external deps."""
    def reset(self):
        self.s = (torch.rand(4) - 0.5) * 0.1
        self.n = 0
        return self.s.clone(), {}
    def step(self, a):
        x, xd, th, thd = self.s
        f = 10.0 if a == 1 else -10.0
        ct, st = math.cos(th), math.sin(th)
        tmp = (f + 0.05 * thd ** 2 * st) / 1.1
        thacc = (9.8 * st - ct * tmp) / (0.5 * (4 / 3 - 0.1 * ct ** 2 / 1.1))
        xacc = tmp - 0.05 * thacc * ct / 1.1
        dt = 0.02
        self.s = torch.tensor([x + dt * xd, xd + dt * xacc, th + dt * thd, thd + dt * thacc])
        self.n += 1
        done = bool(abs(self.s[0]) > 2.4 or abs(self.s[2]) > 0.21 or self.n >= 500)
        return self.s.clone(), 1.0, done, False, {}

try:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    print("using gymnasium CartPole-v1")
except Exception:
    env = MiniCartPole()
    print("gymnasium not found — using built-in MiniCartPole")

def as_t(obs):
    return obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
'''),
        md('''
### Strategy A — REINFORCE: make good episodes more likely

The policy network outputs action *probabilities*. Play an episode, then nudge the policy so
that actions from **high-reward** episodes become more likely. The gradient of
`-log π(a) · return` does exactly that.

One subtlety: rewards should be **discounted** — an action deserves credit mostly for rewards
that follow *soon* after it.
'''),
        code('''
def discount(rewards, gamma=0.99):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.insert(0, g)
    out = torch.tensor(out)
    return (out - out.mean()) / (out.std() + 1e-8)     # normalize: huge variance reduction

policy = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

R_hist = []
for ep in range(250):
    obs, _ = env.reset()
    logps, rewards = [], []
    done = False
    while not done:
        dist = torch.distributions.Categorical(logits=policy(as_t(obs)))
        a = dist.sample()
        logps.append(dist.log_prob(a))
        obs, r, done, trunc, _ = env.step(a.item())
        rewards.append(r); done = done or trunc
    loss = -(torch.stack(logps) * discount(rewards)).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    R_hist.append(sum(rewards))
    if ep % 50 == 0:
        print(f"episode {ep:3d}: reward {R_hist[-1]:.0f}  (avg last 25: {sum(R_hist[-25:]) / len(R_hist[-25:]):.0f})")

plt.plot(R_hist, alpha=0.4)
plt.plot(torch.tensor(R_hist).unfold(0, 20, 1).mean(1), lw=2)
plt.title("REINFORCE learning to balance"); plt.xlabel("episode"); plt.ylabel("reward"); plt.show()
'''),
        md('''
### Strategy B — DQN: learn the *value* of actions

Instead of a policy, learn **Q(state, action)** = expected future reward — then act greedily.
Two stabilizers made this work on Atari in 2015:
- a **replay buffer** (learn from shuffled past experience, not just the last step),
- a frozen **target network** (chase a stable target, not your own tail).
'''),
        code('''
from collections import deque

qnet = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
target.load_state_dict(qnet.state_dict())
opt = torch.optim.Adam(qnet.parameters(), lr=1e-3)
buf = deque(maxlen=8000)
eps, GAMMA = 1.0, 0.99

D_hist = []
for ep in range(160):
    obs, _ = env.reset()
    done, total = False, 0.0
    while not done:
        if random.random() < eps:
            a = random.randrange(2)
        else:
            with torch.no_grad():
                a = qnet(as_t(obs)).argmax().item()
        nxt, r, done, trunc, _ = env.step(a); done = done or trunc
        buf.append((as_t(obs), a, r, as_t(nxt), float(done)))
        obs, total = nxt, total + r

        if len(buf) >= 256:
            batch = random.sample(buf, 128)
            s, a_, r_, s2, d = map(lambda z: torch.stack(z) if torch.is_tensor(z[0]) else torch.tensor(z), zip(*batch))
            q = qnet(s).gather(1, a_.long().unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = r_ + GAMMA * target(s2).max(1).values * (1 - d)
            opt.zero_grad(); F.smooth_l1_loss(q, tgt).backward(); opt.step()

    eps = max(0.05, eps * 0.96)
    if ep % 15 == 0:
        target.load_state_dict(qnet.state_dict())
    D_hist.append(total)
    if ep % 40 == 0:
        print(f"episode {ep:3d}: reward {total:.0f}  epsilon {eps:.2f}")

plt.plot(D_hist, alpha=0.4)
plt.plot(torch.tensor(D_hist).unfold(0, 15, 1).mean(1), lw=2)
plt.title("DQN learning CartPole"); plt.xlabel("episode"); plt.ylabel("reward"); plt.show()
'''),
        md('''
Two philosophies, one result: an agent that balances a pole having never been told how.
**Policy-based** (REINFORCE → PPO → RLHF, which aligns chatbots) and **value-based**
(DQN → Rainbow) are the two great families of RL.
'''),
        registry_cell('''
_register("warmup", 5,
    lambda s: "reward" in s.lower(),
    "the only feedback an RL agent ever receives")
_register("discount_fn", 20,
    lambda f: (lambda o: len(o) == 3 and abs(o[0] - 2.9701) < 1e-3 and abs(o[1] - 1.99) < 1e-3 and abs(o[2] - 1.0) < 1e-9)(f([1.0, 1.0, 1.0], 0.99)),
    "raw discounted returns, NO normalization: g_t = r_t + gamma * g_{t+1}. discount([1,1,1], .99) -> [2.9701, 1.99, 1.0]")
_register("greedy_eps", 15,
    lambda f: 0.25 < sum(f([0.0, 5.0, 1.0], 0.9) != 1 for _ in range(500)) / 500 < 0.75,
    "def act(q, eps): return random action with prob eps, else argmax(q). Test uses eps=0.9")
_register("why_target", 10,
    lambda s: "stab" in s.lower() or "moving" in s.lower() or "chas" in s.lower() or "fixed" in s.lower(),
    "one phrase: why freeze a separate target network in DQN?")
'''),
        code('''
check("warmup", "reward")
'''),
        boss_md([
            '`discount_fn` (20 XP) — write raw (unnormalized) discounted returns; submit the function.',
            '`greedy_eps` (15 XP) — write `act(q_values, eps)` doing epsilon-greedy selection; submit the function.',
            '`why_target` (10 XP) — why does DQN need a target network?',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("12_art_of_action.ipynb", cells)


# ---------------------------------------------------------------------------
# 13 — The Art of Speed
# ---------------------------------------------------------------------------
def _q13():
    cells = [
        header("13_art_of_speed.ipynb", "Act III", "13", "The Art of Speed",
               "Measure first, then make it fast: profiling, compile, precision, quantization, and shipping.",
               prev="12_art_of_action.ipynb", nxt="14_final_boss.ipynb"),
        setup_cell(),
        md('''
## The optimizer's oath: *measure first*

Guessing at bottlenecks wastes weeks. Build a timing habit before touching any knob.
'''),
        code('''
import time

def stopwatch(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()          # GPU work is async — always sync before timing!
    t0 = time.time()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000

net = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 256)).to(device)
x = torch.randn(256, 512, device=device)
base_ms = stopwatch(lambda: net(x))
print(f"baseline forward: {base_ms:.2f} ms")
'''),
        md('''
### Weapon 1 — batching (the free lunch)

The single biggest inference speedup most people never take: process many inputs per call.
'''),
        code('''
one = torch.randn(1, 512, device=device)
t_single = stopwatch(lambda: net(one))
t_batched = stopwatch(lambda: net(x))       # 256 at once
print(f"256 items one-by-one: ~{t_single * 256:7.1f} ms")
print(f"256 items in a batch:  {t_batched:7.2f} ms   ({t_single * 256 / t_batched:.0f}x faster)")
'''),
        md('''
### Weapon 2 — `torch.compile` (PyTorch 2.x)

A JIT that fuses ops and generates optimized kernels. One line. Gains are largest on GPU;
on Windows/CPU it may fall back gracefully — which is why we guard it.
'''),
        code('''
try:
    fast = torch.compile(net)
    fast(x)                                  # first call compiles (slow), later calls are fast
    comp_ms = stopwatch(lambda: fast(x))
    print(f"eager: {base_ms:.2f} ms  |  compiled: {comp_ms:.2f} ms")
except Exception as e:
    print(f"torch.compile unavailable here ({type(e).__name__}) — try it on Colab GPU, where it shines.")
'''),
        md('''
### Weapon 3 — mixed precision (GPU only)

float16/bfloat16 halves memory and can double throughput on tensor cores. The recipe (for Colab):

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x).sum()
scaler.scale(loss).backward()
scaler.step(opt); scaler.update()
```
'''),
        code('''
if device.type == "cuda":
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y_half = net(x)
    print("autocast ran; output dtype inside:", y_half.dtype)
else:
    print("CPU session — AMP demo skipped (run this cell on a Colab GPU).")
'''),
        md('''
### Weapon 4 — quantization: int8 weights for CPU serving

`quantize_dynamic` converts `Linear` weights to int8 at load time — smaller files, faster CPU
matmuls, tiny accuracy cost.
'''),
        code('''
import os
net_cpu = net.cpu().eval()
try:
    q8 = torch.quantization.quantize_dynamic(net_cpu, {nn.Linear}, dtype=torch.qint8)

    def size_mb(m):
        torch.save(m.state_dict(), "_tmp.pt"); s = os.path.getsize("_tmp.pt") / 1e6; os.remove("_tmp.pt"); return s

    xc = x.cpu()
    print(f"size:  fp32 {size_mb(net_cpu):.2f} MB  ->  int8 {size_mb(q8):.2f} MB")
    print(f"speed: fp32 {stopwatch(lambda: net_cpu(xc)):.2f} ms  ->  int8 {stopwatch(lambda: q8(xc)):.2f} ms")
    print(f"max output drift: {(net_cpu(xc) - q8(xc)).abs().max().item():.4f}")
except Exception as e:
    print(f"quantization unavailable ({type(e).__name__})")
'''),
        md('''
### Weapon 5 — shipping: TorchScript

`torch.jit.trace` freezes the model into a self-contained artifact loadable from C++, mobile,
or a Python process with no model class definition — the standard handoff to production.
'''),
        code('''
os.makedirs("checkpoints", exist_ok=True)
example = torch.randn(1, 512)
scripted = torch.jit.trace(net_cpu, example)
scripted.save("checkpoints/speednet.pt")

revived = torch.jit.load("checkpoints/speednet.pt")
print("round-trip output identical:", torch.allclose(net_cpu(example), revived(example)))

class Predictor:
    """What your web service would import — no model class needed."""
    def __init__(self, path):
        self.m = torch.jit.load(path); self.m.eval()
    @torch.no_grad()
    def __call__(self, feats):
        return self.m(torch.as_tensor(feats, dtype=torch.float32).reshape(1, -1))

p = Predictor("checkpoints/speednet.pt")
print("served prediction shape:", p([0.0] * 512).shape)
'''),
        registry_cell('''
_register("warmup", 5,
    lambda s: "sync" in s.lower() or "async" in s.lower(),
    "why must you call torch.cuda.synchronize() before timing GPU code? (mention sync/async)")
_register("quant_shrink", 15,
    lambda sizes: len(sizes) == 2 and sizes[1] < 0.5 * sizes[0],
    "quantize any Linear-heavy model; submit (fp32_mb, int8_mb) — int8 should be < half the size")
_register("script_match", 15,
    lambda ok: ok is True,
    "trace any model, save, reload, and submit torch.allclose(original(x), reloaded(x))")
_register("first_move", 10,
    lambda s: "measure" in s.lower() or "profile" in s.lower() or "benchmark" in s.lower() or "time" in s.lower(),
    "one word: what do you ALWAYS do before optimizing?")
'''),
        code('''
check("first_move", "measure")
'''),
        boss_md([
            '`warmup` (5 XP) — why synchronize before timing GPU code?',
            '`quant_shrink` (15 XP) — quantize a model; submit `(fp32_mb, int8_mb)`.',
            '`script_match` (15 XP) — TorchScript round-trip; submit the `allclose` bool.',
        ]),
        code('''
# ⚔️ your attempts here...

# xp_report()
'''),
    ]
    write_notebook("13_art_of_speed.ipynb", cells)


# ---------------------------------------------------------------------------
# 14 — The Final Boss
# ---------------------------------------------------------------------------
def _q14():
    cells = [
        header("14_final_boss.ipynb", "Act III", "14", "👾 The Final Boss",
               "Everything you've forged, in one battle: data → model → evaluation → robustness → shipping.",
               prev="13_art_of_speed.ipynb", nxt="15_beyond_the_forge_onnx.ipynb"),
        setup_cell(),
        md('''
## The battle

The boss has **5 health bars**. Each phase you clear knocks one out. Clear all five and you've
demonstrated end-to-end mastery: you can take a problem from raw data to a shipped artifact.

| Phase | Objective | Check |
|-------|-----------|-------|
| 1 · Supply lines | augmented train/test loaders | `phase1` |
| 2 · The weapon | model ≥ **94%** test accuracy | `phase2` |
| 3 · Battle damage | ≥ **75%** accuracy under heavy noise | `phase3` |
| 4 · Know your enemy | identify the most-confused glyph pair | `phase4` |
| 5 · Victory lap | TorchScript artifact that matches the model | `phase5` |

A **reference walkthrough** lives at the bottom — one honorable attempt before peeking.
'''),
        code(GLYPH_DATA),
        code('''
from torch.utils.data import Dataset, DataLoader

# The battlefield: fixed test set (never train on it!), and a harder noisy variant
X_test, y_test = make_glyphs(n_per_class=100, seed=999)
X_noisy = (X_test + 0.35 * torch.randn_like(X_test)).clamp(0, 1)

def test_acc(model, X=X_test, y=y_test):
    model.eval()
    with torch.no_grad():
        return (model(X.to(device)).argmax(1).cpu() == y).float().mean().item()
'''),
        registry_cell('''
_BOSS = ["phase1", "phase2", "phase3", "phase4", "phase5"]
def boss_health():
    down = sum(1 for p in _BOSS if p in _XP["done"])
    hearts = "💜" * (5 - down) + "💥" * down
    print(f"👾 BOSS HEALTH: {hearts}   ({5 - down}/5 remaining)")
    if down == 5:
        print("🏆 THE BOSS FALLS. You are a PyTorch practitioner. Course complete!")

_register("phase1", 20,
    lambda tr, te: hasattr(tr, "batch_size") and hasattr(te, "batch_size")
                   and next(iter(tr))[0].shape[1:] == (1, 20, 20)
                   and len(tr.dataset) >= 800,
    "two DataLoaders over glyph data (>=800 train samples), batched, images shaped (1,20,20)")
_register("phase2", 30,
    lambda m: test_acc(m) >= 0.94,
    "train a CNN on your loaders until test accuracy >= 94% (augmentation + a few epochs does it)")
_register("phase3", 20,
    lambda m: test_acc(m, X_noisy, y_test) >= 0.75,
    "robustness: >= 75% on the noisy test set — train WITH noise augmentation")
_register("phase4", 15,
    lambda pair: set(map(str.lower, pair)) == {"cross", "slash"},
    "compute the confusion matrix on X_test; which two glyphs are confused most? submit like ('cross','slash')")
_register("phase5", 15,
    lambda ok: ok is True,
    "torch.jit.trace your model, save+reload, submit torch.allclose(model(x), reloaded(x)) for a test batch")
'''),
        md("## ⚔️ Your battle — phases 1 through 5"),
        code('''
# Phase 1 — forge your data pipeline
# ...
# check("phase1", train_loader, test_loader); boss_health()
'''),
        code('''
# Phase 2 — forge and train your weapon
# ...
# check("phase2", model); boss_health()
'''),
        code('''
# Phase 3 — survive the noise storm
# check("phase3", model); boss_health()

# Phase 4 — study the confusion matrix
# check("phase4", ("?", "?")); boss_health()

# Phase 5 — ship it
# check("phase5", matches); boss_health()
'''),
        md('''
---
## 🎓 Reference walkthrough (sensei's battle)

Every cell below runs — and defeats the boss. Compare it with your approach afterwards.
'''),
        code('''
# Phase 1 — supply lines: augmented training data
class BattleGlyphs(Dataset):
    def __init__(self, n=1600, noise_aug=0.30, seed=7):
        self.X, self.y = make_glyphs(n // 4, seed=seed)
        self.noise_aug = noise_aug
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        img, lbl = self.X[i], self.y[i]
        if random.random() < 0.5:
            img = img.flip(-1)
        img = img.roll(random.randint(-1, 1), dims=-1)
        img = (img + self.noise_aug * random.random() * torch.randn_like(img)).clamp(0, 1)
        return img, lbl

train_loader = DataLoader(BattleGlyphs(), batch_size=64, shuffle=True)
test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=256)
check("phase1", train_loader, test_loader)
boss_health()
'''),
        code('''
# Phase 2 — the weapon: a compact CNN with batchnorm
class BossSlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4),
        )
    def forward(self, x):
        return self.net(x)

model = BossSlayer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
for epoch in range(6):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()
    print(f"epoch {epoch}: clean test {test_acc(model)*100:.1f}% | noisy test {test_acc(model, X_noisy, y_test)*100:.1f}%")
check("phase2", model)
boss_health()
'''),
        code('''
# Phase 3 — robustness came free with noise augmentation
check("phase3", model)
boss_health()
'''),
        code('''
# Phase 4 — know your enemy: the confusion matrix
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(1).cpu()
cm = torch.zeros(4, 4, dtype=torch.int)
for t, p in zip(y_test, preds):
    cm[t, p] += 1

off = cm.clone(); off.fill_diagonal_(0)
i, j = divmod(int((off + off.T).argmax()), 4)
print("confusion matrix:\\n", cm)
print(f"most-confused pair: {GLYPHS[i]} <-> {GLYPHS[j]}  (they share a diagonal stroke!)")
check("phase4", (GLYPHS[i], GLYPHS[j]))
boss_health()
'''),
        code('''
# Phase 5 — victory lap: ship a TorchScript artifact
import os
os.makedirs("checkpoints", exist_ok=True)
model.cpu().eval()
example = X_test[:4]
torch.jit.trace(model, example).save("checkpoints/boss_slayer.pt")
revived = torch.jit.load("checkpoints/boss_slayer.pt")
matches = torch.allclose(model(example), revived(example), atol=1e-5)
check("phase5", matches)
boss_health()
model.to(device)

xp_report()
'''),
        md('''
---
# 🏆 Course complete

Look how far you've come:

- **Act I** — you *built* an autograd engine, then recognized it inside PyTorch.
- **Act II** — you gave networks eyes, memory, and attention. You built a GPT. You earned a
  debugging black belt.
- **Act III** — you sculpted hearts from noise, taught an agent to balance a pole, and shipped
  optimized artifacts.

### Where to next
- 🖥️ Re-run quests 07, 09, 11 on a **Colab GPU** with the scale knobs cranked up.
- 🕹️ Revisit **the Arcade** (`streamlit run arcade/Home.py`) — the demos will feel different now
  that you know what's underneath.
- 🌍 Pick a dataset you *care about* and repeat the Final Boss pipeline on it. That's the real
  final exam — and the beginning of your own work.

⚒️ *The forge is yours now.*
'''),
    ]
    write_notebook("14_final_boss.ipynb", cells)
