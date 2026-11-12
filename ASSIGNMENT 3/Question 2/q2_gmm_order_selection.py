"""
EECE5644 Fall 2026 - Assignment 3, Question 2
GMM model-order selection with 10-fold cross-validation (log-likelihood criterion).

Pipeline
--------
1. True pdf: a 4-component 2-D GMM with different means, covariances and weights,
   where components 1 and 2 overlap significantly.
2. For each repetition r = 1..R and each dataset size N in {10, 100, 1000}:
     - draw N iid samples from the true GMM
     - 10-fold CV: for every candidate order K = 1..10, fit a GMM by EM (maximum
       likelihood) on the 9 training folds, and score the held-out fold with its
       log-likelihood. Sum held-out log-likelihoods over all folds.
     - selected order = argmax_K (total validation log-likelihood)
3. Report how often each order is selected for each N (tables + figures).

Run:  python q2_gmm_order_selection.py            (default: 100 repetitions)
      python q2_gmm_order_selection.py --reps 200 --jobs -1
      python q2_gmm_order_selection.py --quick      (fast smoke test)
Results are written to a "results_q2" folder next to this script.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 1. True GMM (4 components, 2-D)
# ----------------------------------------------------------------------------
TRUE_WEIGHTS = np.array([0.20, 0.35, 0.15, 0.30])
TRUE_MEANS = np.array([[0.0, 0.0],
                       [2.0, 1.0],   # overlaps heavily with component 1
                       [8.0, 0.0],
                       [4.0, 7.0]])
TRUE_COVS = np.array([[[2.0, 0.5], [0.5, 1.0]],
                      [[1.0, -0.3], [-0.3, 2.0]],
                      [[1.5, 0.0], [0.0, 0.5]],
                      [[1.0, 0.4], [0.4, 1.2]]])
TRUE_K = 4

SIZES = [10, 100, 1000]
ORDERS = list(range(1, 11))


def sample_gmm(n, rng):
    """Draw n iid samples: pick a component by its weight, then sample that Gaussian."""
    labels = rng.choice(TRUE_K, size=n, p=TRUE_WEIGHTS)
    X = np.empty((n, 2))
    for k in range(TRUE_K):
        idx = labels == k
        if idx.any():
            X[idx] = rng.multivariate_normal(TRUE_MEANS[k], TRUE_COVS[k], size=idx.sum())
    return X, labels


def true_logpdf(X):
    comps = np.column_stack([np.log(w) + multivariate_normal(m, C).logpdf(X)
                             for w, m, C in zip(TRUE_WEIGHTS, TRUE_MEANS, TRUE_COVS)])
    return logsumexp(comps, axis=1)


# ----------------------------------------------------------------------------
# 2. K-fold cross-validation of GMM orders
# ----------------------------------------------------------------------------
def cv_loglik(X, n_folds, n_init, reg_covar, seed):
    """Return average held-out log-likelihood per sample for each order in ORDERS.

    Training: EM (maximum likelihood) on the training folds, best of n_init restarts.
    Validation: sum of log p(x) of the held-out samples under the fitted GMM.
    An order is marked -inf if it cannot be fitted (K > number of training samples).
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    n = len(X)
    kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
    total = np.zeros(len(ORDERS))
    for f, (tr, va) in enumerate(kf.split(X)):
        for j, K in enumerate(ORDERS):
            if not np.isfinite(total[j]):
                continue
            if K > len(tr):            # EM cannot fit more components than points
                total[j] = -np.inf
                continue
            gmm = GaussianMixture(n_components=K, covariance_type="full",
                                  n_init=n_init, init_params="kmeans",
                                  max_iter=500, tol=1e-4, reg_covar=reg_covar,
                                  random_state=seed + 1000 * f + j)
            try:
                gmm.fit(X[tr])
                total[j] += gmm.score_samples(X[va]).sum()
            except Exception:
                total[j] = -np.inf
    return total / n


def run_one_rep(rep, base_seed, n_folds, n_init, reg_covar):
    rng = np.random.default_rng([base_seed, rep])
    out = []
    for N in SIZES:
        X, _ = sample_gmm(N, rng)
        seed = int(rng.integers(0, 2**31 - 10**6))
        out.append(cv_loglik(X, n_folds, n_init, reg_covar, seed))
    return np.array(out)          # shape (len(SIZES), len(ORDERS))


# ----------------------------------------------------------------------------
# 3. Plotting helpers
# ----------------------------------------------------------------------------
def draw_ellipse(ax, mean, cov, nsig=2.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False, **kw))


def plot_true_gmm(out_dir, base_seed):
    rng = np.random.default_rng([base_seed, 99999])
    xs, ys = np.meshgrid(np.linspace(-5, 12, 300), np.linspace(-5, 11, 300))
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    pdf = np.exp(true_logpdf(grid)).reshape(xs.shape)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ax, N in zip(axes, SIZES):
        X, lab = sample_gmm(N, rng)
        ax.contour(xs, ys, pdf, levels=12, cmap="Greys", linewidths=0.7)
        for k in range(TRUE_K):
            ax.scatter(X[lab == k, 0], X[lab == k, 1], s=12 if N < 1000 else 5,
                       color=colors[k], alpha=0.8, label=f"comp {k+1} (w={TRUE_WEIGHTS[k]:.2f})")
            draw_ellipse(ax, TRUE_MEANS[k], TRUE_COVS[k], color=colors[k], lw=2)
        ax.set_title(f"True GMM with one sample dataset, N={N}")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_aspect("equal"); ax.set_xlim(-5, 12); ax.set_ylim(-5, 11)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_true_gmm_and_samples.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_selection_rates(rates, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.27
    x = np.array(ORDERS)
    for i, N in enumerate(SIZES):
        bars = ax.bar(x + (i - 1) * width, 100 * rates[i], width, label=f"N = {N}")
        for b, v in zip(bars, rates[i]):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, 100 * v + 0.8, f"{100*v:.0f}",
                        ha="center", fontsize=7)
    ax.axvline(TRUE_K, color="k", ls="--", lw=1)
    ax.text(TRUE_K + 0.05, ax.get_ylim()[1] * 0.95, "true order", fontsize=9)
    ax.set_xticks(ORDERS)
    ax.set_xlabel("Selected GMM order K")
    ax.set_ylabel("Selection rate (%)")
    ax.set_title("Rate at which each GMM order is selected by 10-fold CV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_selection_rates_bar.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(100 * rates, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    for i in range(len(SIZES)):
        for j in range(len(ORDERS)):
            v = 100 * rates[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < 60 else "black", fontsize=9)
    ax.set_xticks(range(len(ORDERS))); ax.set_xticklabels(ORDERS)
    ax.set_yticks(range(len(SIZES))); ax.set_yticklabels([f"N={N}" for N in SIZES])
    ax.set_xlabel("Selected GMM order K")
    ax.set_title("Selection rate (%) heatmap")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_selection_rates_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_validation_curves(ll, ref, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (ax, N) in enumerate(zip(axes, SIZES)):
        L = np.where(np.isfinite(ll[i]), ll[i], np.nan)       # (reps, orders)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # K=10 infeasible for N=10
            med = np.nanmedian(L, axis=0)
            q25, q75 = np.nanpercentile(L, 25, axis=0), np.nanpercentile(L, 75, axis=0)
        ax.fill_between(ORDERS, q25, q75, alpha=0.3, label="25-75 percentile")
        ax.plot(ORDERS, med, "o-", label="median over repetitions")
        ax.axhline(ref, color="r", ls="--", label=f"true pdf E[log p(x)] = {ref:.2f}")
        ax.axvline(TRUE_K, color="k", ls=":", lw=1)
        best = np.nanmax(med)
        ax.set_ylim(best - max(4.0, 3 * (ref - best)), ref + 0.5)
        ax.set_xticks(ORDERS)
        ax.set_xlabel("GMM order K")
        ax.set_ylabel("avg validation log-likelihood per sample")
        ax.set_title(f"N = {N}  (values below axis are clipped)")
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_validation_loglik_vs_order.png"), dpi=150)
    plt.close(fig)


def plot_example_fits(out_dir, base_seed, n_folds, n_init, reg_covar):
    """For one example dataset of each size: run CV, fit the chosen order on all data, plot."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rng = np.random.default_rng([base_seed, 12345])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, N in zip(axes, SIZES):
        X, _ = sample_gmm(N, rng)
        ll = cv_loglik(X, n_folds, n_init, reg_covar, seed=7)
        K = ORDERS[int(np.argmax(ll))]
        gmm = GaussianMixture(K, covariance_type="full", n_init=n_init,
                              reg_covar=reg_covar, random_state=0).fit(X)
        ax.scatter(X[:, 0], X[:, 1], s=10 if N < 1000 else 4, color="grey", alpha=0.6)
        for k in range(TRUE_K):
            draw_ellipse(ax, TRUE_MEANS[k], TRUE_COVS[k], color="k", ls="--", lw=1.5)
        for k in range(K):
            draw_ellipse(ax, gmm.means_[k], gmm.covariances_[k], color="tab:red", lw=2)
            ax.plot(*gmm.means_[k], "rx")
        ax.set_title(f"N={N}: CV selected K={K}\n(black dashed = true, red = fitted, 2-sigma)")
        ax.set_aspect("equal"); ax.set_xlim(-5, 12); ax.set_ylim(-5, 11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_example_selected_fits.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=100, help="number of repetitions of the experiment")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--n-init", type=int, default=5, help="EM random restarts per fit")
    p.add_argument("--reg-covar", type=float, default=1e-6, help="added to covariance diagonals")
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    p.add_argument("--seed", type=int, default=5644)
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "results_q2"))
    p.add_argument("--quick", action="store_true", help="smoke test: 4 reps, 1 init")
    a = p.parse_args()
    if a.quick:
        a.reps, a.n_init = 4, 1
    os.makedirs(a.out, exist_ok=True)

    # --- describe true GMM / overlap check ---
    d12 = np.linalg.norm(TRUE_MEANS[0] - TRUE_MEANS[1])
    avg_eig = [np.trace(C) / 2 for C in TRUE_COVS]
    ref = true_logpdf(sample_gmm(200000, np.random.default_rng(1))[0]).mean()
    print("True GMM")
    for k in range(TRUE_K):
        print(f"  comp {k+1}: w={TRUE_WEIGHTS[k]:.2f}  mean={TRUE_MEANS[k]}  "
              f"cov={TRUE_COVS[k].tolist()}  avg eigenvalue={avg_eig[k]:.2f}")
    print(f"  overlap check comps 1&2: ||m1-m2|| = {d12:.3f}  vs  "
          f"avg-eig1 + avg-eig2 = {avg_eig[0] + avg_eig[1]:.3f}")
    print(f"  E[log p(x)] under true pdf (MC, 2e5 samples) = {ref:.4f}\n")

    # --- main experiment ---
    t0 = time.time()
    print(f"Running {a.reps} repetitions x sizes {SIZES} x orders 1..10 x {a.folds} folds "
          f"(n_init={a.n_init}) ...")
    res = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(run_one_rep)(r, a.seed, a.folds, a.n_init, a.reg_covar) for r in range(a.reps))
    ll = np.stack(res, axis=1)          # (sizes, reps, orders)
    print(f"done in {time.time() - t0:.1f} s\n")

    selected = np.array(ORDERS)[np.argmax(ll, axis=2)]      # (sizes, reps)
    counts = np.array([[np.sum(selected[i] == K) for K in ORDERS] for i in range(len(SIZES))])
    rates = counts / a.reps

    # --- tables ---
    hdr = "| N | " + " | ".join(f"K={K}" for K in ORDERS) + " | mode | mean | median | P(select true K=4) |"
    sep = "|" + "---|" * (len(ORDERS) + 5)
    lines = [f"Selection rate (%) over {a.reps} repetitions", "", hdr, sep]
    for i, N in enumerate(SIZES):
        s = selected[i]
        mode = ORDERS[int(np.argmax(counts[i]))]
        lines.append(f"| {N} | " + " | ".join(f"{100*r:.0f}" for r in rates[i]) +
                     f" | {mode} | {s.mean():.2f} | {np.median(s):.0f} | {100*rates[i][TRUE_K-1]:.0f}% |")
    lines += ["", "Median (over repetitions) of average validation log-likelihood per sample", "",
              "| N | " + " | ".join(f"K={K}" for K in ORDERS) + " |", "|" + "---|" * (len(ORDERS) + 1)]
    for i, N in enumerate(SIZES):
        L = np.where(np.isfinite(ll[i]), ll[i], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # K=10 infeasible for N=10
            med = np.nanmedian(L, axis=0)
        lines.append(f"| {N} | " + " | ".join("infeasible" if np.isnan(m) else f"{m:.3f}" for m in med) + " |")
    lines.append(f"\nReference: E[log p(x)] under the true pdf = {ref:.4f}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.out, "results_tables.md"), "w") as f:
        f.write(table + "\n")
    np.savez(os.path.join(a.out, "raw_results.npz"), ll=ll, selected=selected,
             sizes=SIZES, orders=ORDERS)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    # --- figures ---
    plot_true_gmm(a.out, a.seed)
    plot_selection_rates(rates, a.out)
    plot_validation_curves(ll, ref, a.out)
    plot_example_fits(a.out, a.seed, a.folds, a.n_init, a.reg_covar)


if __name__ == "__main__":
    main()
