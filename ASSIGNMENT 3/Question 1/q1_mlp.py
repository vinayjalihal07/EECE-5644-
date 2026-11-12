"""
MLP approximation of class posteriors + MAP classification, with 10-fold CV model-order selection.

Usage:
    python q1_mlp.py            # full run
    python q1_mlp.py --quick    # fast smoke test (fewer P values / restarts)

Outputs (written to ./results_q1/):
    results.json              all numbers (theoretical P(error), CV curves, chosen P, test P(error))
    results_table.csv         summary table for the report
    pe_vs_ntrain.png          test P(error) vs training-set size (semilog-x) + optimal line
    cv_curves.png             10-fold CV P(error) vs number of perceptrons, one curve per training set
    data_scatter.png          3-D scatter of a training set (visual check of the distribution)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------------------------
# 1. True data distribution: C=4 classes, uniform priors, 3-D Gaussian class-conditionals
# ----------------------------------------------------------------------------------------------
C = 4
PRIORS = np.full(C, 1.0 / C)
SCALE = 1.1  # mean separation; tuned so that the true-pdf MAP classifier has ~15% P(error)
MEANS = SCALE * np.array([[1, 1, 1],
                          [1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, 1]], dtype=float)  # vertices of a regular tetrahedron
COVS = np.array([[[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]],
                 [[1.5, -0.4, 0.2], [-0.4, 0.8, 0.0], [0.2, 0.0, 1.2]],
                 [[0.8, 0.0, 0.3], [0.0, 1.4, -0.3], [0.3, -0.3, 1.0]],
                 [[1.2, 0.2, -0.4], [0.2, 1.2, 0.3], [-0.4, 0.3, 0.9]]])

N_TRAIN_LIST = [100, 500, 1000, 5000, 10000]
N_TEST = 100000


def generate_data(n, rng):
    """Draw labels from the priors, then x | label from the class-conditional Gaussian."""
    y = rng.choice(C, size=n, p=PRIORS)
    X = np.empty((n, 3))
    for k in range(C):
        idx = y == k
        X[idx] = rng.multivariate_normal(MEANS[k], COVS[k], size=idx.sum())
    return X, y


def map_classify_true_pdf(X):
    """Theoretically optimal (min-P(error)) rule: argmax_k log p(x|k) + log P(k)."""
    log_post = np.stack([mvn(MEANS[k], COVS[k]).logpdf(X) + np.log(PRIORS[k]) for k in range(C)], axis=1)
    return log_post.argmax(axis=1)


# ----------------------------------------------------------------------------------------------
# 2. MLP: 3 -> P (ELU, smooth ramp) -> 4 (softmax)
# ----------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.hidden = nn.Linear(3, P)
        self.act = nn.ELU()
        self.out = nn.Linear(P, C)

    def forward(self, x):
        # Returns logits. softmax(logits) are the posterior estimates. CrossEntropyLoss applies
        # log-softmax internally, so minimizing it == maximizing the log-likelihood of the
        # softmax outputs (done this way for numerical stability).
        return self.out(self.act(self.hidden(x)))

    def posteriors(self, x):
        return torch.softmax(self.forward(x), dim=1)


def train_mlp(X, y, P, n_restarts, seed, max_iter=500):
    """ML training (min average cross-entropy) with full-batch L-BFGS.
    Repeats from n_restarts random initializations and keeps the model with the highest
    training-data log-likelihood (i.e. lowest training cross-entropy)."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()  # mean of -log softmax(logits)[y]
    best_model, best_loss = None, np.inf
    for r in range(n_restarts):
        torch.manual_seed(seed + 1000 * r)
        model = MLP(P)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-7, tolerance_change=1e-10,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            final_loss = loss_fn(model(Xt), yt).item()
        if np.isfinite(final_loss) and final_loss < best_loss:
            best_loss, best_model = final_loss, model
    return best_model, best_loss


def mlp_error(model, X, y):
    """MAP rule with the MLP posteriors: decide argmax_k P_hat(k|x); return empirical P(error)."""
    with torch.no_grad():
        post = model.posteriors(torch.as_tensor(X, dtype=torch.float32)).numpy()
    return float(np.mean(post.argmax(axis=1) != y))


# ----------------------------------------------------------------------------------------------
# 3. Model order selection: 10-fold CV, objective = validation P(error)
# ----------------------------------------------------------------------------------------------
def cross_validate(X, y, P_list, K, n_restarts, seed):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    mean_err = []
    for P in P_list:
        errs = []
        for f, (tr, va) in enumerate(folds):
            model, _ = train_mlp(X[tr], y[tr], P, n_restarts, seed=seed + 17 * f + P)
            errs.append(mlp_error(model, X[va], y[va]))
        mean_err.append(float(np.mean(errs)))
        print(f"    P={P:3d}  CV P(error)={mean_err[-1]:.4f}")
    # pick the P with min CV error; on ties prefer the smaller (simpler) network
    best_P = P_list[int(np.argmin(mean_err))]
    return best_P, mean_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_q1")
    args = ap.parse_args()

    if args.quick:
        P_list, cv_restarts, final_restarts = [1, 4, 16], 1, 2
    else:
        P_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
        cv_restarts, final_restarts = 3, 10
    K = 10

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- data ----
    train_sets = {n: generate_data(n, rng) for n in N_TRAIN_LIST}
    X_test, y_test = generate_data(N_TEST, rng)
    print("Test-set class counts:", np.bincount(y_test, minlength=C))
    # sanity check: sample mean/cov of the generated test data should match the true parameters
    for k in range(C):
        Xk = X_test[y_test == k]
        print(f"  class {k + 1}: max|mean err|={np.abs(Xk.mean(0) - MEANS[k]).max():.3f}  "
              f"max|cov err|={np.abs(np.cov(Xk.T) - COVS[k]).max():.3f}")

    # ---- theoretically optimal classifier ----
    pe_opt = float(np.mean(map_classify_true_pdf(X_test) != y_test))
    print(f"\nTheoretically optimal (true-pdf MAP) test P(error) = {pe_opt:.4f}")

    results = {"P_list": P_list, "pe_optimal": pe_opt, "per_n": {}}
    for n in N_TRAIN_LIST:
        X, y = train_sets[n]
        print(f"\n=== N_train = {n} ===  (class counts {np.bincount(y, minlength=C)})")
        print("  10-fold CV:")
        best_P, cv_err = cross_validate(X, y, P_list, K, cv_restarts, seed=args.seed + n)
        print(f"  -> selected P = {best_P}")

        model, train_ce = train_mlp(X, y, best_P, final_restarts, seed=args.seed + 7 * n)
        pe_train = mlp_error(model, X, y)
        pe_test = mlp_error(model, X_test, y_test)
        print(f"  final model: train CE={train_ce:.4f}  train P(error)={pe_train:.4f}  "
              f"TEST P(error)={pe_test:.4f}   [elapsed {time.time() - t0:.0f}s]")
        results["per_n"][str(n)] = {"cv_error": cv_err, "best_P": best_P,
                                    "train_cross_entropy": train_ce,
                                    "train_error": pe_train, "test_error": pe_test}

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary table ----
    lines = ["N_train,selected_P,min_CV_P(error),train_P(error),test_P(error),test_minus_optimal"]
    print("\n" + "-" * 78)
    print(f"{'N_train':>8} {'P*':>4} {'CV Pe':>8} {'train Pe':>9} {'test Pe':>8} {'gap to opt':>11}")
    for n in N_TRAIN_LIST:
        r = results["per_n"][str(n)]
        gap = r["test_error"] - pe_opt
        print(f"{n:>8} {r['best_P']:>4} {min(r['cv_error']):>8.4f} {r['train_error']:>9.4f} "
              f"{r['test_error']:>8.4f} {gap:>11.4f}")
        lines.append(f"{n},{r['best_P']},{min(r['cv_error']):.4f},{r['train_error']:.4f},"
                     f"{r['test_error']:.4f},{gap:.4f}")
    print(f"{'optimal':>8} {'-':>4} {'-':>8} {'-':>9} {pe_opt:>8.4f}")
    lines.append(f"optimal,-,-,-,{pe_opt:.4f},0")
    with open(os.path.join(args.out, "results_table.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- plot: test P(error) vs N_train ----
    ns = np.array(N_TRAIN_LIST)
    pes = [results["per_n"][str(n)]["test_error"] for n in N_TRAIN_LIST]
    Ps = [results["per_n"][str(n)]["best_P"] for n in N_TRAIN_LIST]
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(ns, pes, "o-", label="Trained MLP (MAP with MLP posteriors)")
    for n, pe, P in zip(ns, pes, Ps):
        plt.annotate(f"P={P}", (n, pe), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.axhline(pe_opt, color="r", ls="--", label=f"Theoretically optimal = {pe_opt:.4f}")
    plt.xlabel("Number of training samples")
    plt.ylabel("Empirical test P(error)  (100k samples)")
    plt.title("Test P(error) vs training-set size")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "pe_vs_ntrain.png"), dpi=150)

    # ---- plot: CV curves ----
    plt.figure(figsize=(7, 4.5))
    for n in N_TRAIN_LIST:
        r = results["per_n"][str(n)]
        plt.semilogx(P_list, r["cv_error"], "o-", ms=4, label=f"N={n} (P*={r['best_P']})")
    plt.axhline(pe_opt, color="k", ls="--", lw=1, label="Optimal")
    plt.xlabel("Number of hidden perceptrons P")
    plt.ylabel("10-fold CV P(error)")
    plt.title("Model order selection by 10-fold cross-validation")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "cv_curves.png"), dpi=150)

    # ---- plot: data scatter ----
    X, y = train_sets[1000]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    for k in range(C):
        ax.scatter(*X[y == k].T, s=6, label=f"class {k + 1}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title("Training data (N=1000)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "data_scatter.png"), dpi=150)

    print(f"\nDone in {time.time() - t0:.0f}s. Results in ./{args.out}/")


if __name__ == "__main__":
    main()
