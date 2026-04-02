# debug_geodesic_convergence.py
# Run from the same directory as your main vae.py
# Usage: python debug_geodesic_convergence.py
#
# Loads a 2-decoder model, picks a random image pair, and plots
# the geodesic energy over optimization steps so you can verify convergence.

import torch
import torch.nn as nn
import torch.distributions as td
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
from torchvision import datasets, transforms

# ── Copy the model classes verbatim from your vae.py ──────────────────────────

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std  = nn.Parameter(torch.ones(M),  requires_grad=False)
    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net
    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(std)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_nets):
        super().__init__()
        self.decoder_nets = nn.ModuleList(decoder_nets)
        self._step_count = 0  # ← needed to match saved model structure
    def forward(self, z):
        if self.training:
            idx = self._step_count % len(self.decoder_nets)
            self._step_count += 1
            means = self.decoder_nets[idx](z)
        else:
            means = sum(net(z) for net in self.decoder_nets) / len(self.decoder_nets)
        return td.Independent(td.Normal(means, 1e-1), 3)

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior   = prior
        self.decoder = decoder
        self.encoder = encoder
    def elbo(self, x):
        q = self.encoder(x); z = q.rsample()
        return torch.mean(self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z))
    def forward(self, x):
        return -self.elbo(x)

def new_encoder(M):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.Softmax(), nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.Softmax(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.Flatten(), nn.Linear(512, 2*M),
    )

def new_decoder(M):
    return nn.Sequential(
        nn.Linear(M, 512), nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(), nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(), nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(), nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )

# ── Energy function (efficient: pre-decode once) ──────────────────────────────

def compute_ensemble_energy(model, z_curve):
    """Monte Carlo approximation: draw one (l,k) pair per segment."""
    num_decoders = len(model.decoder.decoder_nets)
    # Pre-decode the full curve once per decoder
    all_decoded = [net(z_curve) for net in model.decoder.decoder_nets]

    energy = torch.tensor(0.0, device=z_curve.device)
    for i in range(len(z_curve) - 1):
        l = torch.randint(0, num_decoders, (1,)).item()
        k = torch.randint(0, num_decoders, (1,)).item()
        energy = energy + ((all_decoded[l][i] - all_decoded[k][i+1])**2).sum()
    return energy

def compute_geodesic_length(model, z_curve):
    """Arc-length: sum of sqrt(segment_energy) using exact expectation."""
    num_decoders = len(model.decoder.decoder_nets)
    with torch.no_grad():
        all_decoded = torch.stack([net(z_curve) for net in model.decoder.decoder_nets])
    length = 0.0
    for i in range(len(z_curve) - 1):
        seg = torch.tensor(0.0, device=all_decoded.device)
        for l in range(num_decoders):
            for k in range(num_decoders):
                seg = seg + ((all_decoded[l, i] - all_decoded[k, i+1])**2).sum()
        seg = seg / (num_decoders ** 2)
        length += math.sqrt(max(seg.item(), 0.0))
    return length

# ── Config ────────────────────────────────────────────────────────────────────

EXPERIMENT_FOLDER = "test/model_dec3_run1"   # ← adjust if needed
M          = 2
NUM_DEC    = 3
NUM_T      = 30        # curve resolution
NUM_STEPS  = 2000      # optimization steps
LR         = 5e-3      # try 1e-3 and compare to 1e-5
LOG_EVERY  = 10        # record energy every N steps
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load data ─────────────────────────────────────────────────────────────────

def subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data    = data[idx][:num_data].unsqueeze(1).float() / 255
    new_targets = targets[idx][:num_data]
    return torch.utils.data.TensorDataset(new_data, new_targets)

test_raw = datasets.MNIST("data/", train=False, download=True,
                           transform=transforms.ToTensor())
test_data = subsample(test_raw.data, test_raw.targets, 2048, 3)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# ── Load model ────────────────────────────────────────────────────────────────

model = VAE(
    GaussianPrior(M),
    GaussianDecoder([new_decoder(M) for _ in range(NUM_DEC)]),
    GaussianEncoder(new_encoder(M)),
).to(DEVICE)
model.load_state_dict(torch.load(f"{EXPERIMENT_FOLDER}/model.pt", map_location=DEVICE))
model.eval()
print(f"Loaded model from {EXPERIMENT_FOLDER}")

# ── Pick a random image pair and encode ───────────────────────────────────────

torch.manual_seed(SEED)
test_images = torch.cat([x for x, _ in test_loader], dim=0).to(DEVICE)

idx = torch.randint(0, len(test_images), (2,))
x_start = test_images[idx[0]:idx[0]+1]
x_end   = test_images[idx[1]:idx[1]+1]

with torch.no_grad():
    z_start = model.encoder(x_start).mean.detach()
    z_end   = model.encoder(x_end).mean.detach()

euc_dist = torch.norm(z_start - z_end).item()
print(f"z_start: {z_start.cpu().numpy()}")
print(f"z_end:   {z_end.cpu().numpy()}")
print(f"Euclidean distance: {euc_dist:.4f}")

# ── Initialise straight-line curve ────────────────────────────────────────────

t = torch.linspace(0, 1, NUM_T).view(-1, 1).to(DEVICE)
z_curve_init  = ((1 - t) * z_start + t * z_end).detach()
z_intermediate = z_curve_init[1:-1].clone().detach().requires_grad_(True)

init_energy = compute_ensemble_energy(model, z_curve_init).item()
print(f"\nInitial energy: {init_energy:.4f}")

# ── Optimize and record energy log ────────────────────────────────────────────

optimizer  = torch.optim.Adam([z_intermediate], lr=LR)
energy_log = []   # (step, energy)

print(f"\nOptimizing for {NUM_STEPS} steps at lr={LR}...")
for step in range(NUM_STEPS):
    optimizer.zero_grad()
    z_full = torch.cat([z_start, z_intermediate, z_end], dim=0)
    loss   = compute_ensemble_energy(model, z_full)
    loss.backward()
    optimizer.step()

    if step % LOG_EVERY == 0:
        energy_log.append((step, loss.item()))
        if step % 100 == 0:
            print(f"  step {step:4d}  energy = {loss.item():.4f}")

# Final curve
z_final = torch.cat([z_start, z_intermediate, z_end], dim=0).detach()
final_energy = compute_ensemble_energy(model, z_final).item()
print(f"\nFinal energy:   {final_energy:.4f}")
print(f"Energy reduced: {(1 - final_energy/init_energy)*100:.1f}%")

# Arc-lengths
init_length  = compute_geodesic_length(model, z_curve_init)
final_length = compute_geodesic_length(model, z_final)
print(f"\nInitial geodesic length: {init_length:.4f}")
print(f"Final   geodesic length: {final_length:.4f}")
print(f"Euclidean distance:      {euc_dist:.4f}")
assert final_length >= euc_dist * 0.95, \
    "WARNING: geodesic shorter than Euclidean — something is wrong!"

# ── Background latent scatter ─────────────────────────────────────────────────

all_z, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        all_z.append(model.encoder(x.to(DEVICE)).mean.cpu())
        all_labels.append(y)
all_z      = torch.cat(all_z).numpy()
all_labels = torch.cat(all_labels).numpy()

# ── Plot ──────────────────────────────────────────────────────────────────────

steps, energies = zip(*energy_log)

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# --- Panel 1: Energy over steps ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(steps, energies, color="#e63946", linewidth=1.8, label="MC energy")
ax1.axhline(init_energy,  color="gray",    linestyle="--", linewidth=1, label=f"Init  {init_energy:.2f}")
ax1.axhline(final_energy, color="#2a9d8f", linestyle="--", linewidth=1, label=f"Final {final_energy:.2f}")
ax1.set_xlabel("Optimisation step")
ax1.set_ylabel("Curve energy")
ax1.set_title("Geodesic Energy vs. Step")
ax1.legend(fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.5)

# --- Panel 2: Energy on log scale (reveals convergence rate) ---
ax2 = fig.add_subplot(gs[1])
ax2.semilogy(steps, energies, color="#e63946", linewidth=1.8)
ax2.set_xlabel("Optimisation step")
ax2.set_ylabel("Curve energy  (log scale)")
ax2.set_title("Log-scale Energy (convergence rate)")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

# --- Panel 3: Latent space with both curves ---
ax3 = fig.add_subplot(gs[2])
scatter = ax3.scatter(all_z[:, 0], all_z[:, 1], c=all_labels,
                      cmap="viridis", alpha=0.4, s=10)
plt.colorbar(scatter, ax=ax3, label="Digit class")

init_np  = z_curve_init.cpu().numpy()
final_np = z_final.cpu().numpy()

ax3.plot(init_np[:,  0], init_np[:,  1], color="gray",    linestyle="--",
         linewidth=1.5, alpha=0.7, label="Straight line")
ax3.plot(final_np[:, 0], final_np[:, 1], color="#e63946", linestyle="-",
         linewidth=2.5, alpha=0.9, label="Optimised geodesic")
ax3.scatter(*z_start.cpu().numpy()[0], marker="*", s=200, color="blue",  zorder=5, label="Start")
ax3.scatter(*z_end.cpu().numpy()[0],   marker="*", s=200, color="green", zorder=5, label="End")

ax3.set_xlabel("z₁"); ax3.set_ylabel("z₂")
ax3.set_title("Latent Space  (2 decoders)")
ax3.legend(fontsize=7)
ax3.grid(True, linestyle="--", alpha=0.4)

# --- Shared title ---
fig.suptitle(
    f"Geodesic Convergence Debug  |  2 decoders  |  lr={LR}  |  {NUM_STEPS} steps\n"
    f"Energy {init_energy:.3f} → {final_energy:.3f}  "
    f"({(1-final_energy/init_energy)*100:.1f}% reduction)  |  "
    f"Geo length {final_length:.3f}  vs  Euclidean {euc_dist:.3f}",
    fontsize=10
)

out_path = f"{EXPERIMENT_FOLDER}/debug_convergence_lr{LR}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to: {out_path}")