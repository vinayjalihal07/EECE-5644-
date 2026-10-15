import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from matplotlib.lines import Line2D

# ============================================================
# EECE 5644 Fall 2026 - Assignment 1 - Question 2
# Fixed random seed for reproducibility
# ============================================================
SEED = 42
N = 10000
rng = np.random.default_rng(SEED)

# Class priors
priors = np.array([0.3, 0.3, 0.4])

# Four Gaussian components
mu1  = np.array([0.0, 0.0, 0.0])
mu2  = np.array([2.4, 0.4, 0.0])
mu3a = np.array([1.0, 2.2, 0.3])
mu3b = np.array([1.2, 0.8, 2.2])

C1 = np.array([
    [1.0,  0.2,  0.1],
    [0.2,  0.8,  0.1],
    [0.1,  0.1,  0.9]
])

C2 = np.array([
    [0.9, -0.2,  0.0],
    [-0.2, 1.0,  0.15],
    [0.0,  0.15, 0.8]
])

C3a = np.array([
    [0.8,  0.1, -0.1],
    [0.1,  0.9,  0.2],
    [-0.1, 0.2,  1.0]
])

C3b = np.array([
    [1.0,  0.15,  0.05],
    [0.15, 0.8,  -0.15],
    [0.05,-0.15,  0.9]
])

# ------------------------------------------------------------
# Generate 10,000 samples from the class prior distribution
# ------------------------------------------------------------
L = rng.choice([1, 2, 3], size=N, p=priors)
X = np.zeros((N, 3))
component = np.zeros(N, dtype=int)

for n, label in enumerate(L):
    if label == 1:
        X[n] = rng.multivariate_normal(mu1, C1)
        component[n] = 1
    elif label == 2:
        X[n] = rng.multivariate_normal(mu2, C2)
        component[n] = 2
    else:
        # Equal-weight mixture for class 3
        if rng.random() < 0.5:
            X[n] = rng.multivariate_normal(mu3a, C3a)
            component[n] = 3
        else:
            X[n] = rng.multivariate_normal(mu3b, C3b)
            component[n] = 4

# Save the exact generated data
np.savetxt(
    "q2_generated_data.csv",
    np.column_stack((X, L, component)),
    delimiter=",",
    header="x1,x2,x3,true_label,gaussian_component",
    comments=""
)

# ------------------------------------------------------------
# True class-conditional PDFs
# ------------------------------------------------------------
# p(x|L=1) = N(mu1,C1)
# p(x|L=2) = N(mu2,C2)
# p(x|L=3) = 0.5*N(mu3a,C3a) + 0.5*N(mu3b,C3b)
log_p1 = multivariate_normal.logpdf(X, mean=mu1, cov=C1)
log_p2 = multivariate_normal.logpdf(X, mean=mu2, cov=C2)
log_g3a = multivariate_normal.logpdf(X, mean=mu3a, cov=C3a)
log_g3b = multivariate_normal.logpdf(X, mean=mu3b, cov=C3b)

log_p3 = logsumexp(
    np.vstack([
        np.log(0.5) + log_g3a,
        np.log(0.5) + log_g3b
    ]),
    axis=0
)

# log joint score log[P(L=j)p(x|L=j)]
log_joint = np.column_stack([
    np.log(priors[0]) + log_p1,
    np.log(priors[1]) + log_p2,
    np.log(priors[2]) + log_p3
])

# Posterior probabilities
log_evidence = logsumexp(log_joint, axis=1)
posterior = np.exp(log_joint - log_evidence[:, None])

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def confusion_matrix(D, L):
    """
    Rows = decision i
    Columns = true class j
    Returns:
      count[i,j]
      P(D=i | L=j)
    """
    counts = np.zeros((3, 3), dtype=int)
    for i in range(1, 4):
        for j in range(1, 4):
            counts[i-1, j-1] = np.sum((D == i) & (L == j))

    true_counts = np.array([np.sum(L == j) for j in [1,2,3]])
    probs = counts / true_counts[None, :]
    return counts, probs

def expected_risk_from_confusion(P, Lambda):
    """
    R = sum_j P(L=j) sum_i lambda_ij P(D=i | L=j)
    """
    R = 0.0
    for j in range(3):
        for i in range(3):
            R += priors[j] * Lambda[i,j] * P[i,j]
    return R

def plot_classification(D, title, filename):
    correct = (D == L)
    markers = {1: ".", 2: "o", 3: "^"}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in [1,2,3]:
        for ok in [True, False]:
            mask = (L == label) & (correct == ok)
            ax.scatter(
                X[mask,0], X[mask,1], X[mask,2],
                marker=markers[label],
                c="green" if ok else "red",
                s=12,
                alpha=0.55
            )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(title)

    legend_classes = [
        Line2D([0],[0], marker=".", color="none", markerfacecolor="black",
               markeredgecolor="black", markersize=10, label="True class 1"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="black",
               markeredgecolor="black", markersize=7, label="True class 2"),
        Line2D([0],[0], marker="^", color="none", markerfacecolor="black",
               markeredgecolor="black", markersize=7, label="True class 3")
    ]

    legend_correctness = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="green",
               markeredgecolor="green", markersize=7, label="Correct"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="red",
               markeredgecolor="red", markersize=7, label="Incorrect")
    ]

    leg1 = ax.legend(handles=legend_classes, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=legend_correctness, loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.show()

# ============================================================
# PART A: MAP / Minimum probability of error
# ============================================================
D_map = np.argmax(log_joint, axis=1) + 1
C_map, P_map = confusion_matrix(D_map, L)
Perror_map = np.mean(D_map != L)

print("\n================ PART A: MAP ================")
print("Class counts:", [np.sum(L==j) for j in [1,2,3]])
print("\nCount matrix (rows=decision, columns=true):")
print(C_map)
print("\nConfusion matrix P(D=i|L=j):")
print(P_map)
print("\nEmpirical P(error) =", Perror_map)

plot_classification(
    D_map,
    "Question 2 Part A: MAP Classification",
    "q2_partA_MAP_3D.png"
)

# ============================================================
# PART B: ERM with asymmetric loss matrices
# ============================================================
Lambda10 = np.array([
    [0, 1, 10],
    [1, 0, 10],
    [1, 1, 0]
], dtype=float)

Lambda100 = np.array([
    [0, 1, 100],
    [1, 0, 100],
    [1, 1, 0]
], dtype=float)

def run_erm(Lambda, name, plot_filename):
    # R(i|x) = sum_j lambda_ij P(L=j|x)
    conditional_risk = posterior @ Lambda.T
    D = np.argmin(conditional_risk, axis=1) + 1

    counts, probs = confusion_matrix(D, L)
    estimated_risk = expected_risk_from_confusion(probs, Lambda)
    sample_average_loss = np.mean(Lambda[D-1, L-1])
    p_error = np.mean(D != L)

    print(f"\n================ {name} ================")
    print("\nCount matrix (rows=decision, columns=true):")
    print(counts)
    print("\nConfusion matrix P(D=i|L=j):")
    print(probs)
    print("\nEstimated expected risk using true priors =", estimated_risk)
    print("Raw sample-average loss =", sample_average_loss)
    print("Empirical P(error) =", p_error)

    plot_classification(D, name, plot_filename)

    return D, counts, probs, estimated_risk, sample_average_loss, p_error

run_erm(
    Lambda10,
    "Question 2 Part B: ERM with Lambda_10",
    "q2_partB_Lambda10_3D.png"
)

run_erm(
    Lambda100,
    "Question 2 Part B: ERM with Lambda_100",
    "q2_partB_Lambda100_3D.png"
)
