"""

Part 1: theoretically optimal (min-P(error)) classifier, ROC curve, P(error) estimate.
Part 2: ML-trained logistic-linear and logistic-quadratic posterior approximations.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

SEED = 5644
rng = np.random.default_rng(SEED)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------- True data distribution ----------------
PRIORS = np.array([0.6, 0.4])                      # P(L=0), P(L=1)
WEIGHTS = np.array([0.5, 0.5])                     # w_i1, w_i2
MEANS = {0: np.array([[-0.9, -1.1], [0.8, 0.75]]),  # m01, m02
         1: np.array([[-1.1, 0.9], [0.9, -0.75]])}  # m11, m12
COV = np.diag([0.75, 1.25])                        # C_ij for all ij


def generate_data(N):
    """Draw N iid (x, L) pairs: label from priors, then mixture component, then Gaussian."""
    labels = (rng.random(N) < PRIORS[1]).astype(int)
    comps = (rng.random(N) < WEIGHTS[1]).astype(int)
    X = np.zeros((N, 2))
    for l in (0, 1):
        for c in (0, 1):
            idx = (labels == l) & (comps == c)
            X[idx] = rng.multivariate_normal(MEANS[l][c], COV, idx.sum())
    return X, labels


def class_conditional_pdf(X, l):
    return sum(WEIGHTS[c] * multivariate_normal.pdf(X, MEANS[l][c], COV) for c in (0, 1))


def optimal_score(X):
    """Discriminant score: log-likelihood ratio ln p(x|L=1) - ln p(x|L=0)."""
    return np.log(class_conditional_pdf(X, 1)) - np.log(class_conditional_pdf(X, 0))


# ---------------- Datasets ----------------
datasets = {name: generate_data(N) for name, N in
            [("D20_train", 20), ("D200_train", 200), ("D2000_train", 2000), ("D10K_validate", 10000)]}
X_val, y_val = datasets["D10K_validate"]
for name, (X, y) in datasets.items():
    print(f"{name:14s}: N={len(y):5d}, N0={np.sum(y == 0):5d}, N1={np.sum(y == 1):5d}")


# ---------------- Part 1: optimal classifier + ROC ----------------
def roc_curve(scores, labels):
    """Sweep threshold tau over all scores (decide 1 if score >= tau). Returns FPR, TPR, error counts, taus."""
    order = np.argsort(-scores)
    s_sorted, l_sorted = scores[order], labels[order]
    N1, N0 = labels.sum(), len(labels) - labels.sum()
    tp = np.concatenate(([0], np.cumsum(l_sorted)))
    fp = np.concatenate(([0], np.cumsum(1 - l_sorted)))
    taus = np.concatenate(([np.inf], s_sorted))
    errors = fp + (N1 - tp)
    return fp / N0, tp / N1, errors, taus


def confusion(decisions, labels):
    tpr = np.mean(decisions[labels == 1] == 1)
    fpr = np.mean(decisions[labels == 0] == 1)
    p_err = np.mean(decisions != labels)
    return fpr, tpr, p_err


def count_pairs(decisions, labels):
    """Counts of (decision, true label) pairs: TN=(0,0), FP=(1,0), FN=(0,1), TP=(1,1)."""
    return {f"{n}": int(np.sum((decisions == d) & (labels == l)))
            for n, d, l in (("TN", 0, 0), ("FP", 1, 0), ("FN", 0, 1), ("TP", 1, 1))}


scores_val = optimal_score(X_val)
gamma = PRIORS[0] / PRIORS[1]
tau_theory = np.log(gamma)
D_opt = (scores_val > tau_theory).astype(int)
fpr_opt, tpr_opt, perr_opt = confusion(D_opt, y_val)

fpr, tpr, errors, taus = roc_curve(scores_val, y_val)
k_emp = np.argmin(errors)
print("\nPart 1: optimal classifier")
print(f"  theoretical threshold gamma = {gamma:.3f}, ln(gamma) = {tau_theory:.4f}")
print(f"  at ln(gamma): FPR={fpr_opt:.4f}, TPR={tpr_opt:.4f}, P(error)={perr_opt:.4f} "
      f"({np.sum(D_opt != y_val)} errors / {len(y_val)})")
print(f"  decision-truth counts: {count_pairs(D_opt, y_val)}")
print(f"  at empirical best threshold: FPR={fpr[np.argmin(errors)]:.4f}, TPR={tpr[np.argmin(errors)]:.4f}")
print(f"  empirical best threshold on D10K: tau={taus[k_emp]:.4f} (gamma={np.exp(taus[k_emp]):.3f}), "
      f"P(error)={errors[k_emp] / len(y_val):.4f}")

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(fpr, tpr, lw=2, label="ROC (log-likelihood-ratio score)")
ax.plot(fpr_opt, tpr_opt, "r*", ms=16,
        label=f"min-P(error) classifier, $\\gamma$={gamma:.2f}\nP(error)={perr_opt:.4f}")
ax.plot(fpr[k_emp], tpr[k_emp], "ko", mfc="none", ms=11,
        label=f"empirical min, $\\gamma$={np.exp(taus[k_emp]):.2f}\nP(error)={errors[k_emp] / len(y_val):.4f}")
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set_xlabel("False positive rate  P(D=1 | L=0)")
ax.set_ylabel("True positive rate  P(D=1 | L=1)")
ax.set_title("ROC curve of the optimal classifier on $D^{10K}_{validate}$")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/q1_part1_roc.png", dpi=150)

# Grid used for all decision-boundary plots
g1, g2 = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5.5, 5.5, 400))
grid = np.column_stack([g1.ravel(), g2.ravel()])


def plot_decisions(ax, X, y, score_grid, tau, s=4):
    """Shade decision regions (D=0 blue, D=1 orange), draw the boundary, and scatter samples by true label."""
    ax.contourf(g1, g2, score_grid, levels=[-np.inf, tau, np.inf], colors=["#d6e6f5", "#fde3c8"])
    ax.contour(g1, g2, score_grid, levels=[tau], colors="k", linewidths=2)
    ax.scatter(*X[y == 0].T, s=s, c="tab:blue", marker="o", alpha=0.6, label="L=0")
    ax.scatter(*X[y == 1].T, s=s, c="tab:orange", marker="^", alpha=0.6, label="L=1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")


opt_grid = optimal_score(grid).reshape(g1.shape)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
plot_decisions(ax, X_val, y_val, opt_grid, tau_theory)
ax.plot([], [], "k-", lw=2, label="decision boundary")
ax.set_title(f"Optimal decision boundary on $D^{{10K}}_{{validate}}$, P(error)={perr_opt:.4f}")
ax.legend(loc="upper right", fontsize=8, markerscale=2)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/q1_part1_boundary.png", dpi=150)


# ---------------- Part 2: logistic posterior approximations ----------------
def features(X, kind):
    x1, x2 = X[:, 0], X[:, 1]
    one = np.ones(len(X))
    if kind == "linear":
        return np.column_stack([one, x1, x2])
    return np.column_stack([one, x1, x2, x1 ** 2, x1 * x2, x2 ** 2])


def nll(w, Z, y):
    """Negative log-likelihood of labels under h = sigmoid(w^T z), averaged over samples.
    -[y ln h + (1-y) ln(1-h)] = ln(1 + e^a) - y a, with a = w^T z (stable via logaddexp)."""
    a = Z @ w
    return np.mean(np.logaddexp(0, a) - y * a)


def nll_grad(w, Z, y):
    a = Z @ w
    return Z.T @ (1 / (1 + np.exp(-a)) - y) / len(y)


def train_logistic(X, y, kind):
    Z = features(X, kind)
    res = minimize(nll, np.zeros(Z.shape[1]), args=(Z, y), jac=nll_grad, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": 10000})
    return res.x, res


results = {}
for kind in ("linear", "quadratic"):
    print(f"\nPart 2: logistic-{kind}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for j, name in enumerate(("D20_train", "D200_train", "D2000_train")):
        X_tr, y_tr = datasets[name]
        w, res = train_logistic(X_tr, y_tr, kind)
        D_tr = (features(X_tr, kind) @ w >= 0).astype(int)
        D_val = (features(X_val, kind) @ w >= 0).astype(int)
        perr_tr = np.mean(D_tr != y_tr)
        perr_val = np.mean(D_val != y_val)
        results[(kind, name)] = perr_val
        print(f"  {name:12s}: w = {np.array2string(w, precision=3, suppress_small=True)}")
        print(f"  {'':12s}  final NLL/N = {res.fun:.4f}, converged={res.success}, "
              f"train P(err)={perr_tr:.4f}, validation P(error)={perr_val:.4f} "
              f"({np.sum(D_val != y_val)} errors)")
        print(f"  {'':12s}  validation decision-truth counts: {count_pairs(D_val, y_val)}")

        boundary = (features(grid, kind) @ w).reshape(g1.shape)
        for i, (X, y, tag, perr, s) in enumerate(((X_tr, y_tr, name, perr_tr, 16),
                                                  (X_val, y_val, "D10K_validate", perr_val, 3))):
            ax = axes[i, j]
            plot_decisions(ax, X, y, boundary, 0, s=s)
            ax.contour(g1, g2, opt_grid, levels=[tau_theory], colors="g", linewidths=1.2, linestyles="--")
            ax.set_title(f"trained on {name}\nshown on {tag}, P(error)={perr:.4f}", fontsize=10)
    axes[0, 0].plot([], [], "k-", lw=2, label="trained boundary")
    axes[0, 0].plot([], [], "g--", lw=1.2, label="optimal boundary")
    axes[0, 0].legend(loc="upper right", fontsize=7, markerscale=1.5)
    fig.suptitle(f"Logistic-{kind} classifiers (top: training set, bottom: validation set)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/q1_part2_logistic_{kind}.png", dpi=130)

print("\nSummary of P(error) on D10K_validate")
print(f"  optimal (true pdf)       : {perr_opt:.4f}")
for kind in ("linear", "quadratic"):
    for name in ("D20_train", "D200_train", "D2000_train"):
        print(f"  logistic-{kind:9s} {name:11s}: {results[(kind, name)]:.4f}")
