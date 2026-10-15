import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from pathlib import Path

OUT = Path(__file__).resolve().parent

SEED = 42
N = 10000
P0 = 0.35
P1 = 0.65

rng = np.random.default_rng(SEED)

m0 = np.array([-0.5, -0.5, -0.5, -0.5], dtype=float)
m1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

C0 = 0.25 * np.array([
    [2.0, -0.5,  0.3, 0.0],
    [-0.5, 1.0, -0.5, 0.0],
    [0.3, -0.5,  1.0, 0.0],
    [0.0,  0.0,  0.0, 2.0]
])

C1 = np.array([
    [1.0,  0.3, -0.2, 0.0],
    [0.3,  2.0,  0.3, 0.0],
    [-0.2, 0.3,  1.0, 0.0],
    [0.0,  0.0,  0.0, 3.0]
])

# Generate one 10,000-sample data set and reuse it for Parts A, B, and C.
L = (rng.random(N) < P1).astype(int)

n0 = int(np.sum(L == 0))
n1 = int(np.sum(L == 1))

X = np.empty((N, 4))
X[L == 0] = rng.multivariate_normal(m0, C0, n0)
X[L == 1] = rng.multivariate_normal(m1, C1, n1)


def log_gaussian_pdf(X, mean, covariance):
    d = X.shape[1]
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    centered = X - mean
    inv_covariance = np.linalg.inv(covariance)
    mahalanobis = np.einsum(
        "...i,ij,...j->...",
        centered,
        inv_covariance,
        centered
    )

    return -0.5 * (
        d * np.log(2 * np.pi)
        + logdet
        + mahalanobis
    )


def roc_from_scores(scores):
    order = np.argsort(scores)[::-1]
    sorted_labels = L[order]
    sorted_scores = scores[order]

    tp = np.concatenate(([0], np.cumsum(sorted_labels == 1)))
    fp = np.concatenate(([0], np.cumsum(sorted_labels == 0)))

    tpr = tp / n1
    fpr = fp / n0

    p_error = P0 * fpr + P1 * (1 - tpr)
    min_index = int(np.argmin(p_error))

    if min_index == 0:
        threshold = np.inf
    elif min_index == N:
        threshold = -np.inf
    else:
        threshold = 0.5 * (
            sorted_scores[min_index - 1]
            + sorted_scores[min_index]
        )

    return fpr, tpr, p_error, min_index, threshold


def classification_metrics(decisions):
    tn = int(np.sum((decisions == 0) & (L == 0)))
    fp = int(np.sum((decisions == 1) & (L == 0)))
    fn = int(np.sum((decisions == 0) & (L == 1)))
    tp = int(np.sum((decisions == 1) & (L == 1)))

    fpr = fp / n0
    tpr = tp / n1
    fnr = fn / n1
    p_error = P0 * fpr + P1 * fnr

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "FPR": fpr,
        "TPR": tpr,
        "FNR": fnr,
        "Perror": p_error
    }


def save_roc(fpr, tpr, min_index, title, filename):
    plt.figure(figsize=(6.5, 5.2))
    plt.plot(fpr, tpr, linewidth=2, label="ROC")
    plt.scatter(
        fpr[min_index],
        tpr[min_index],
        s=75,
        marker="o",
        label="Minimum-error operating point"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Chance line"
    )
    plt.xlabel("False Positive Rate, P(D=1|L=0)")
    plt.ylabel("True Positive Rate, P(D=1|L=1)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=220)
    plt.close()


# ------------------------------------------------------------
# Part A: ERM classifier using the true class-conditional PDFs
# ------------------------------------------------------------

log_lr_true = (
    log_gaussian_pdf(X, m1, C1)
    - log_gaussian_pdf(X, m0, C0)
)

fpr_A, tpr_A, p_error_A, k_A, log_gamma_emp_A = roc_from_scores(log_lr_true)

gamma_emp_A = float(np.exp(log_gamma_emp_A))

# For 0-1 loss:
# gamma* = P(L=0) / P(L=1)
gamma_theory = P0 / P1
log_gamma_theory = float(np.log(gamma_theory))

D_A_emp = (log_lr_true > log_gamma_emp_A).astype(int)
D_A_theory = (log_lr_true > log_gamma_theory).astype(int)

metrics_A_emp = classification_metrics(D_A_emp)
metrics_A_theory = classification_metrics(D_A_theory)

save_roc(
    fpr_A,
    tpr_A,
    k_A,
    "Part A: True-PDF ERM ROC",
    "Q1_PartA_ROC.png"
)


# ------------------------------------------------------------
# Part B: Naive Bayes classifier with diagonal covariances
# ------------------------------------------------------------

C0_NB = np.diag(np.diag(C0))
C1_NB = np.diag(np.diag(C1))

log_lr_NB = (
    log_gaussian_pdf(X, m1, C1_NB)
    - log_gaussian_pdf(X, m0, C0_NB)
)

fpr_B, tpr_B, p_error_B, k_B, log_gamma_emp_B = roc_from_scores(log_lr_NB)

gamma_emp_B = float(np.exp(log_gamma_emp_B))

D_B_emp = (log_lr_NB > log_gamma_emp_B).astype(int)
D_B_theory = (log_lr_NB > log_gamma_theory).astype(int)

metrics_B_emp = classification_metrics(D_B_emp)
metrics_B_theory = classification_metrics(D_B_theory)

save_roc(
    fpr_B,
    tpr_B,
    k_B,
    "Part B: Naive Bayes ROC",
    "Q1_PartB_ROC.png"
)


# ------------------------------------------------------------
# Part C: Fisher Linear Discriminant Analysis
# ------------------------------------------------------------

X0 = X[L == 0]
X1 = X[L == 1]

m0_hat = np.mean(X0, axis=0)
m1_hat = np.mean(X1, axis=0)

C0_hat = np.cov(X0, rowvar=False, bias=False)
C1_hat = np.cov(X1, rowvar=False, bias=False)

m_bar = 0.5 * (m0_hat + m1_hat)

Sw = 0.5 * (C0_hat + C1_hat)

Sb = (
    0.5 * np.outer(m0_hat - m_bar, m0_hat - m_bar)
    + 0.5 * np.outer(m1_hat - m_bar, m1_hat - m_bar)
)

eigenvalues, eigenvectors = la.eigh(Sb, Sw)
w_LDA = eigenvectors[:, np.argmax(eigenvalues)]

if w_LDA @ m1_hat < w_LDA @ m0_hat:
    w_LDA = -w_LDA

w_LDA = w_LDA / np.linalg.norm(w_LDA)

lda_scores = X @ w_LDA

fpr_C, tpr_C, p_error_C, k_C, tau_emp = roc_from_scores(lda_scores)

D_C_emp = (lda_scores > tau_emp).astype(int)
metrics_C_emp = classification_metrics(D_C_emp)

save_roc(
    fpr_C,
    tpr_C,
    k_C,
    "Part C: Fisher LDA ROC",
    "Q1_PartC_ROC.png"
)



# ------------------------------------------------------------
# Combined ROC comparison for Parts A, B, and C
# ------------------------------------------------------------

plt.figure(figsize=(6.5, 5.2))
plt.plot(fpr_A, tpr_A, linewidth=2, label="True-PDF ERM")
plt.plot(fpr_B, tpr_B, linewidth=2, label="Naive Bayes")
plt.plot(fpr_C, tpr_C, linewidth=2, label="Fisher LDA")

plt.scatter(fpr_A[k_A], tpr_A[k_A], marker="o", s=55)
plt.scatter(fpr_B[k_B], tpr_B[k_B], marker="s", s=55)
plt.scatter(fpr_C[k_C], tpr_C[k_C], marker="^", s=55)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Chance line"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Question 1: ROC Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "Q1_ROC_Comparison.png", dpi=220)
plt.close()

# ------------------------------------------------------------
# Save the exact generated data set used for all three parts
# ------------------------------------------------------------

data = pd.DataFrame({
    "sample_id": np.arange(1, N + 1),
    "true_label": L,
    "x1": X[:, 0],
    "x2": X[:, 1],
    "x3": X[:, 2],
    "x4": X[:, 3]
})

data.to_csv(
    OUT / "EECE5644_Q1_data_seed42.csv",
    index=False
)


# ------------------------------------------------------------
# Print only the numerical results needed for the report
# ------------------------------------------------------------

np.set_printoptions(precision=6, suppress=True)

print("EECE5644 Assignment 1 - Question 1")
print(f"Seed = {SEED}")
print(f"Total samples = {N}")
print(f"Class 0 samples = {n0}")
print(f"Class 1 samples = {n1}")

print("\nPART A - TRUE-PDF ERM")
print(f"Theoretical gamma = {gamma_theory:.6f}")
print(f"Empirical minimum-error gamma = {gamma_emp_A:.6f}")
print("Empirical minimum-error metrics:")
print(metrics_A_emp)
print("Metrics at theoretical gamma:")
print(metrics_A_theory)

print("\nPART B - NAIVE BAYES")
print("C0_NB =")
print(C0_NB)
print("C1_NB =")
print(C1_NB)
print(f"Empirical minimum-error gamma = {gamma_emp_B:.6f}")
print("Empirical minimum-error metrics:")
print(metrics_B_emp)
print("Metrics at gamma = P0/P1:")
print(metrics_B_theory)

print("\nPART C - FISHER LDA")
print("Estimated m0 =")
print(m0_hat)
print("Estimated m1 =")
print(m1_hat)
print("Estimated C0 =")
print(C0_hat)
print("Estimated C1 =")
print(C1_hat)
print("Sw =")
print(Sw)
print("Sb =")
print(Sb)
print("w_LDA =")
print(w_LDA)
print(f"Empirical minimum-error tau = {tau_emp:.6f}")
print("Empirical minimum-error metrics:")
print(metrics_C_emp)

print("\nGenerated files:")
print("EECE5644_Q1_data_seed42.csv")
print("Q1_PartA_ROC.png")
print("Q1_PartB_ROC.png")
print("Q1_PartC_ROC.png")
print("Q1_ROC_Comparison.png")
