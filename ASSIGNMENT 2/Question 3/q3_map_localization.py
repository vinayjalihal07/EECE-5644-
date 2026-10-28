"""
MAP estimation of a 2-D vehicle position from K noisy range measurements.

Objective (to be minimized over candidate position [x, y]):
    J(x, y) = sum_i (r_i - ||[x,y] - [x_i,y_i]||)^2 / sigma_i^2 + x^2/sigma_x^2 + y^2/sigma_y^2

Outputs (written to ./figures):
    map_contours_K1..K4.png  - individual contour plots
    map_contours_all.png     - 2x2 panel of all K with identical contour levels
    mc_error_vs_K.png        - Monte Carlo localization error / spread vs K

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(5644)
os.makedirs("figures", exist_ok=True)

# ---------------- parameters ----------------
SIGMA_R = 0.3                 # range-noise std (same for all landmarks)
SIGMA_X = SIGMA_Y = 0.25      # prior std about the origin
K_VALUES = [1, 2, 3, 4]
GRID = np.linspace(-2, 2, 401)
# identical contour levels for every K, applied to dJ = J - J_min (log-spaced).
# Subtracting the data-dependent constant J_min does not move the minimizer, and makes
# the levels comparable across K (J_min itself grows with K as residual terms are added).
LEVELS = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128])


def sample_true_position(rng):
    """Uniform sample inside the unit disk."""
    rad = np.sqrt(rng.uniform())
    ang = rng.uniform(0, 2 * np.pi)
    return np.array([rad * np.cos(ang), rad * np.sin(ang)])


def landmarks(K):
    """K evenly spaced landmarks on the unit circle (first one at angle 0)."""
    ang = 2 * np.pi * np.arange(K) / K
    return np.column_stack([np.cos(ang), np.sin(ang)])


def measure(p_true, lm, sigma_r, rng):
    """Range measurements r_i = d_Ti + n_i, resampling any negative value."""
    d = np.linalg.norm(lm - p_true, axis=1)
    r = d + sigma_r * rng.standard_normal(len(d))
    while np.any(r < 0):
        neg = r < 0
        r[neg] = d[neg] + sigma_r * rng.standard_normal(neg.sum())
    return r


def map_objective(X, Y, lm, r, sigma_r, sx, sy):
    """Simplified MAP objective J evaluated on arrays X, Y (any matching shape)."""
    J = X**2 / sx**2 + Y**2 / sy**2
    for (xi, yi), ri in zip(lm, r):
        di = np.sqrt((X - xi) ** 2 + (Y - yi) ** 2)
        J = J + (ri - di) ** 2 / sigma_r**2
    return J


def map_estimate(lm, r, sigma_r, sx, sy):
    """Minimize J: coarse grid search for initialization, then Nelder-Mead refinement."""
    Xg, Yg = np.meshgrid(GRID[::4], GRID[::4])
    Jg = map_objective(Xg, Yg, lm, r, sigma_r, sx, sy)
    k = np.argmin(Jg)
    f = lambda p: map_objective(p[0], p[1], lm, r, sigma_r, sx, sy)
    res = minimize(f, [Xg.flat[k], Yg.flat[k]], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10})
    return res.x, res.fun


def laplace_std(p, lm, r, sigma_r, sx, sy, h=1e-4):
    """Posterior std's from the numerical Hessian of J at p (cov ~ 2 H^-1 since J = -2 log post)."""
    f = lambda q: map_objective(q[0], q[1], lm, r, sigma_r, sx, sy)
    H = np.zeros((2, 2))
    E = np.eye(2) * h
    for a in range(2):
        for b in range(2):
            H[a, b] = (f(p + E[a] + E[b]) - f(p + E[a] - E[b])
                       - f(p - E[a] + E[b]) + f(p - E[a] - E[b])) / (4 * h * h)
    cov = 2 * np.linalg.inv(H)
    return np.sqrt(np.diag(cov))


# ---------------- single realization: contour plots ----------------
p_true = sample_true_position(rng)
print(f"True position: ({p_true[0]:.4f}, {p_true[1]:.4f})")
X, Y = np.meshgrid(GRID, GRID)

fig_all, axes = plt.subplots(2, 2, figsize=(11, 10))
rows = []
for K, ax_all in zip(K_VALUES, axes.flat):
    lm = landmarks(K)
    r = measure(p_true, lm, SIGMA_R, rng)
    J = map_objective(X, Y, lm, r, SIGMA_R, SIGMA_X, SIGMA_Y)
    p_map, J_min = map_estimate(lm, r, SIGMA_R, SIGMA_X, SIGMA_Y)
    std = laplace_std(p_map, lm, r, SIGMA_R, SIGMA_X, SIGMA_Y)
    err = np.linalg.norm(p_map - p_true)
    rows.append((K, r, p_map, err, J_min, std))

    fig_one, ax_one = plt.subplots(figsize=(6, 5.5))
    for ax in (ax_one, ax_all):
        cs = ax.contour(X, Y, J - J_min, levels=LEVELS, cmap="viridis_r", linewidths=1.1)
        ax.clabel(cs, fmt="%g", fontsize=7)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=":", color="gray"))
        ax.plot(lm[:, 0], lm[:, 1], "o", mfc="none", mec="tab:red", ms=10, mew=2,
                label="Landmarks")
        ax.plot(*p_true, "+", color="k", ms=14, mew=2.5, label="True position")
        ax.plot(*p_map, "x", color="tab:orange", ms=10, mew=2.5, label="MAP estimate")
        ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xlabel="x", ylabel="y",
               title=f"K = {K}   (|error| = {err:.3f})")
        ax.legend(loc="upper right", fontsize=8)
    fig_one.tight_layout()
    fig_one.savefig(f"figures/map_contours_K{K}.png", dpi=170)
    plt.close(fig_one)

fig_all.suptitle(r"MAP objective contours $J-J_{min}$, $\sigma_r=0.3$, $\sigma_x=\sigma_y=0.25$ "
                 "(identical levels in all panels)")
fig_all.tight_layout()
fig_all.savefig("figures/map_contours_all.png", dpi=170)
plt.close(fig_all)

print("\nK | ranges | MAP estimate | |error| | J_min | Laplace std (x, y)")
for K, r, p, e, jm, s in rows:
    print(f"{K} | {np.round(r, 3)} | ({p[0]:.3f}, {p[1]:.3f}) | {e:.3f} | "
          f"{jm:.3f} | ({s[0]:.3f}, {s[1]:.3f})")

# ---------------- Monte Carlo: average behavior vs K ----------------
N_MC = 500
mc = {K: {"err": [], "area": []} for K in K_VALUES}
for _ in range(N_MC):
    pt = sample_true_position(rng)
    for K in K_VALUES:
        lm = landmarks(K)
        r = measure(pt, lm, SIGMA_R, rng)
        p, _ = map_estimate(lm, r, SIGMA_R, SIGMA_X, SIGMA_Y)
        s = laplace_std(p, lm, r, SIGMA_R, SIGMA_X, SIGMA_Y)
        mc[K]["err"].append(np.linalg.norm(p - pt))
        mc[K]["area"].append(np.sqrt(s[0] * s[1]))   # geometric-mean posterior std

print(f"\nMonte Carlo over {N_MC} random true positions:")
print("K | mean |error| | median |error| | mean posterior std")
for K in K_VALUES:
    e, a = np.array(mc[K]["err"]), np.array(mc[K]["area"])
    print(f"{K} | {e.mean():.3f} | {np.median(e):.3f} | {a.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot([mc[K]["err"] for K in K_VALUES], tick_labels=K_VALUES, showfliers=False)
ax[0].plot(range(1, 5), [np.mean(mc[K]["err"]) for K in K_VALUES], "o-", label="mean")
ax[0].set(xlabel="K", ylabel=r"$\|\hat{p}_{MAP}-p_T\|$", title="MAP localization error")
ax[0].legend()
ax[1].plot(K_VALUES, [np.mean(mc[K]["area"]) for K in K_VALUES], "s-")
ax[1].set(xlabel="K", ylabel=r"mean $\sqrt{\sigma_{\hat x}\sigma_{\hat y}}$",
          title="Posterior spread (Laplace approx.)", xticks=K_VALUES)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/mc_error_vs_K.png", dpi=170)
plt.close(fig)
