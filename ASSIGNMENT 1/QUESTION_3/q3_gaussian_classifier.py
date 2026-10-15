from pathlib import Path
import argparse
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from sklearn.decomposition import PCA


def load_wine(data_dir: Path):
    path = data_dir / "winequality-white.csv"

    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    df = pd.read_csv(path, sep=";")

    if "quality" not in df.columns:
        raise ValueError("The 'quality' column was not found in the wine dataset.")

    X = df.drop(columns=["quality"]).to_numpy(dtype=np.float64)
    y = df["quality"].to_numpy(dtype=int)
    feature_names = list(df.drop(columns=["quality"]).columns)

    return X, y, feature_names


def load_har(data_dir: Path):
    root = data_dir / "UCI HAR Dataset"

    required_files = [
        root / "train" / "X_train.txt",
        root / "train" / "y_train.txt",
        root / "test" / "X_test.txt",
        root / "test" / "y_test.txt",
    ]

    missing = [p for p in required_files if not p.exists()]

    if missing:
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Could not find the following HAR files:\n"
            f"{missing_text}"
        )

    X_train = np.loadtxt(root / "train" / "X_train.txt", dtype=np.float64)
    y_train = np.loadtxt(root / "train" / "y_train.txt", dtype=int)
    X_test = np.loadtxt(root / "test" / "X_test.txt", dtype=np.float64)
    y_test = np.loadtxt(root / "test" / "y_test.txt", dtype=int)

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    label_names = {}
    labels_path = root / "activity_labels.txt"

    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    label_names[int(parts[0])] = parts[1]

    feature_names = []
    features_path = root / "features.txt"

    if features_path.exists():
        with open(features_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                feature_names.append(parts[1] if len(parts) == 2 else line.strip())

    if not feature_names:
        feature_names = [f"feature_{i + 1}" for i in range(X.shape[1])]

    return X, y, feature_names, label_names


def standardize_for_pca(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    safe_std = std.copy()
    safe_std[safe_std < 1e-12] = 1.0

    return (X - mean) / safe_std


def covariance_sample_average(Xk, mu):
    centered = Xk - mu
    return (centered.T @ centered) / Xk.shape[0]


def covariance_rank_and_scale(C):
    eigvals = np.linalg.eigvalsh(C)

    if eigvals.size == 0:
        return eigvals, 0, 1.0

    max_eig = float(np.max(eigvals))
    tolerance = (
        max(C.shape)
        * np.finfo(np.float64).eps
        * max(max_eig, 1.0)
    )

    nonzero_eigs = eigvals[eigvals > tolerance]
    rank = int(nonzero_eigs.size)
    mean_nonzero_eig = float(np.mean(nonzero_eigs)) if rank > 0 else 1.0

    return eigvals, rank, mean_nonzero_eig


def fit_gaussian_classifier(X, y, alpha=1e-3):
    classes = np.unique(y)
    N, d = X.shape
    models = []

    for cls in classes:
        Xk = X[y == cls]
        Nk = Xk.shape[0]

        prior = Nk / N
        mu = Xk.mean(axis=0)
        C = covariance_sample_average(Xk, mu)

        eigvals, rank, mean_nonzero_eig = covariance_rank_and_scale(C)

        lam = max(alpha * mean_nonzero_eig, 1e-12)
        Creg = C + lam * np.eye(d)

        try:
            L = np.linalg.cholesky(Creg)
        except np.linalg.LinAlgError:
            trial_lam = lam

            for _ in range(10):
                trial_lam *= 10.0
                Creg = C + trial_lam * np.eye(d)

                try:
                    L = np.linalg.cholesky(Creg)
                    lam = trial_lam
                    break
                except np.linalg.LinAlgError:
                    continue
            else:
                raise RuntimeError(
                    f"Could not obtain a positive-definite covariance for class {cls}."
                )

        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        min_raw_eig = float(np.min(eigvals))
        max_raw_eig = float(np.max(eigvals))
        min_reg_eig = min_raw_eig + lam
        max_reg_eig = max_raw_eig + lam

        condition_reg = (
            max_reg_eig
            / max(min_reg_eig, np.finfo(np.float64).eps)
        )

        models.append({
            "class": cls,
            "count": Nk,
            "prior": prior,
            "mean": mu,
            "cov": C,
            "lambda": lam,
            "rank": rank,
            "condition_reg": condition_reg,
            "chol": L,
            "logdet": logdet,
        })

    return models


def predict_gaussian_classifier(X, models, chunk_size=2000):
    N, d = X.shape
    K = len(models)

    scores = np.empty((N, K), dtype=np.float64)
    common_constant = -0.5 * d * np.log(2.0 * np.pi)

    for model_index, model in enumerate(models):
        L = model["chol"]
        mu = model["mean"]
        log_prior = math.log(model["prior"])
        logdet = model["logdet"]

        for start in range(0, N, chunk_size):
            stop = min(start + chunk_size, N)
            difference = X[start:stop] - mu

            transformed = solve_triangular(
                L,
                difference.T,
                lower=True,
                check_finite=False,
                overwrite_b=False,
            )

            mahalanobis_squared = np.sum(transformed * transformed, axis=0)

            scores[start:stop, model_index] = (
                log_prior
                - 0.5 * logdet
                - 0.5 * mahalanobis_squared
                + common_constant
            )

    class_values = np.array([model["class"] for model in models])
    predicted_labels = class_values[np.argmax(scores, axis=1)]

    return predicted_labels, scores


def confusion_matrix_counts(y_true, y_pred, classes):
    class_to_index = {
        cls: index
        for index, cls in enumerate(classes)
    }

    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        row = class_to_index[true_label]
        col = class_to_index[predicted_label]
        cm[row, col] += 1

    return cm


def confusion_matrix_row_probabilities(cm):
    row_sums = cm.sum(axis=1, keepdims=True)

    return np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )


def class_display_names(classes, label_names=None):
    return [
        label_names.get(int(cls), str(cls)) if label_names else str(cls)
        for cls in classes
    ]


def format_confusion_matrix(cm, classes, label_names=None):
    names = class_display_names(classes, label_names)
    df = pd.DataFrame(cm, index=names, columns=names)
    df.index.name = "True \\ Predicted"
    return df.to_string()


def save_confusion_matrix_csv(matrix, classes, path, label_names=None):
    names = class_display_names(classes, label_names)
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.index.name = "True \\ Predicted"
    df.to_csv(path)


def create_pca_2d_plot(
    X_pca,
    y,
    dataset_name,
    output_path,
    label_names=None
):
    pca = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=42
    )

    projected = pca.fit_transform(X_pca)
    classes = np.unique(y)

    plt.figure(figsize=(9, 7))

    for cls in classes:
        mask = y == cls
        class_name = (
            label_names.get(int(cls), str(cls))
            if label_names
            else str(cls)
        )

        plt.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=10,
            alpha=0.55,
            label=class_name,
        )

    explained_percent = pca.explained_variance_ratio_ * 100.0

    plt.xlabel(f"PC1 ({explained_percent[0]:.2f}% variance)")
    plt.ylabel(f"PC2 ({explained_percent[1]:.2f}% variance)")
    plt.title(f"{dataset_name}: PCA 2D Projection")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return pca, projected


def create_pca_3d_plot(
    projected,
    y,
    pca,
    dataset_name,
    output_path,
    label_names=None
):
    classes = np.unique(y)
    explained_percent = pca.explained_variance_ratio_ * 100.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for cls in classes:
        mask = y == cls
        class_name = (
            label_names.get(int(cls), str(cls))
            if label_names
            else str(cls)
        )

        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            projected[mask, 2],
            s=8,
            alpha=0.45,
            label=class_name,
        )

    ax.set_xlabel(f"PC1 ({explained_percent[0]:.2f}%)")
    ax.set_ylabel(f"PC2 ({explained_percent[1]:.2f}%)")
    ax.set_zlabel(f"PC3 ({explained_percent[2]:.2f}%)")
    ax.set_title(f"{dataset_name}: PCA 3D Projection")
    ax.legend(markerscale=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def analyze_dataset(
    dataset_name,
    X_raw,
    y,
    output_dir,
    alpha,
    label_names=None,
):
    print("\n" + "=" * 80)
    print(dataset_name)
    print("=" * 80)

    N, d = X_raw.shape
    classes = np.unique(y)

    unique_classes, counts = np.unique(y, return_counts=True)

    counts_dictionary = {
        int(cls): int(count)
        for cls, count in zip(unique_classes, counts)
    }

    print(f"Samples: {N}")
    print(f"Features: {d}")
    print(f"Encountered classes: {classes.tolist()}")
    print(f"Class counts: {counts_dictionary}")

    fit_start = time.time()

    models = fit_gaussian_classifier(
        X_raw,
        y,
        alpha=alpha
    )

    fit_time = time.time() - fit_start

    print(f"\nRegularization alpha = {alpha:g}")
    print("Per-class covariance diagnostics:")

    print(
        f"{'Class':>22} "
        f"{'N_k':>8} "
        f"{'Prior':>10} "
        f"{'Rank':>8} "
        f"{'Lambda':>14} "
        f"{'Cond(Creg)':>16}"
    )

    for model in models:
        cls = model["class"]

        displayed_class = (
            label_names.get(int(cls), str(cls))
            if label_names
            else str(cls)
        )

        print(
            f"{displayed_class:>22} "
            f"{model['count']:8d} "
            f"{model['prior']:10.6f} "
            f"{model['rank']:8d} "
            f"{model['lambda']:14.6e} "
            f"{model['condition_reg']:16.6e}"
        )

    prediction_start = time.time()

    y_pred, _ = predict_gaussian_classifier(
        X_raw,
        models
    )

    prediction_time = time.time() - prediction_start

    errors = int(np.count_nonzero(y_pred != y))
    p_error = errors / N
    accuracy = 1.0 - p_error

    cm_counts = confusion_matrix_counts(
        y,
        y_pred,
        classes
    )

    cm_probabilities = confusion_matrix_row_probabilities(
        cm_counts
    )

    print("\nClassification results:")
    print(f"Errors = {errors} / {N}")
    print(
        f"Estimated P(error) = "
        f"{p_error:.8f} ({100.0 * p_error:.4f}%)"
    )
    print(
        f"Training accuracy   = "
        f"{accuracy:.8f} ({100.0 * accuracy:.4f}%)"
    )

    print(
        "\nConfusion matrix COUNTS "
        "(rows=true, columns=predicted):"
    )
    print(
        format_confusion_matrix(
            cm_counts,
            classes,
            label_names
        )
    )

    print(
        "\nRow-normalized confusion matrix "
        "P(predicted | true):"
    )
    print(
        format_confusion_matrix(
            np.round(cm_probabilities, 6),
            classes,
            label_names
        )
    )

    print(f"\nFit time: {fit_time:.2f} s")
    print(f"Prediction time: {prediction_time:.2f} s")

    safe_name = (
        dataset_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    counts_path = output_dir / f"{safe_name}_confusion_counts.csv"
    probabilities_path = (
        output_dir / f"{safe_name}_confusion_probabilities.csv"
    )

    save_confusion_matrix_csv(
        cm_counts,
        classes,
        counts_path,
        label_names
    )

    save_confusion_matrix_csv(
        cm_probabilities,
        classes,
        probabilities_path,
        label_names
    )

    print("\nGenerating PCA plots...")

    X_pca = standardize_for_pca(X_raw)

    pca_2d_path = output_dir / f"{safe_name}_pca_2d.png"
    pca_3d_path = output_dir / f"{safe_name}_pca_3d.png"

    pca, projected = create_pca_2d_plot(
        X_pca,
        y,
        dataset_name,
        pca_2d_path,
        label_names
    )

    create_pca_3d_plot(
        projected,
        y,
        pca,
        dataset_name,
        pca_3d_path,
        label_names
    )

    explained = pca.explained_variance_ratio_

    print(
        "PCA explained variance ratios: "
        f"PC1={explained[0]:.6f}, "
        f"PC2={explained[1]:.6f}, "
        f"PC3={explained[2]:.6f}"
    )

    print(f"Saved: {counts_path}")
    print(f"Saved: {probabilities_path}")
    print(f"Saved: {pca_2d_path}")
    print(f"Saved: {pca_3d_path}")

    return {
        "dataset": dataset_name,
        "N": N,
        "d": d,
        "models": models,
        "errors": errors,
        "p_error": p_error,
        "accuracy": accuracy,
        "confusion_counts": cm_counts,
        "confusion_probabilities": cm_probabilities,
        "pca_explained": explained,
    }


def write_summary(results, output_dir, alpha):
    path = output_dir / "q3_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("EECE 5644 - Assignment 1 - Question 3\n")
        f.write("Gaussian Minimum-Probability-of-Error Classifier\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"alpha = {alpha:g}\n\n")

        for result in results:
            f.write("=" * 72 + "\n")
            f.write(f"{result['dataset']}\n")
            f.write("=" * 72 + "\n")

            f.write(f"N = {result['N']}\n")
            f.write(f"d = {result['d']}\n")
            f.write(f"Errors = {result['errors']}\n")
            f.write(
                f"P(error) = {result['p_error']:.8f} "
                f"({100.0 * result['p_error']:.4f}%)\n"
            )
            f.write(
                f"Accuracy = {result['accuracy']:.8f} "
                f"({100.0 * result['accuracy']:.4f}%)\n"
            )

            f.write(
                "\nPCA explained variance ratios:\n"
                f"PC1 = {result['pca_explained'][0]:.8f}\n"
                f"PC2 = {result['pca_explained'][1]:.8f}\n"
                f"PC3 = {result['pca_explained'][2]:.8f}\n"
            )

            f.write("\nPer-class covariance diagnostics:\n")

            for model in result["models"]:
                f.write(
                    f"class={model['class']}, "
                    f"N={model['count']}, "
                    f"prior={model['prior']:.8f}, "
                    f"rank={model['rank']}, "
                    f"lambda={model['lambda']:.8e}, "
                    f"cond(Creg)={model['condition_reg']:.8e}\n"
                )

            f.write(
                "\nConfusion matrix counts "
                "(rows=true, columns=predicted):\n"
            )
            f.write(
                np.array2string(
                    result["confusion_counts"]
                )
            )

            f.write(
                "\n\nRow-normalized confusion matrix "
                "P(predicted|true):\n"
            )
            f.write(
                np.array2string(
                    result["confusion_probabilities"],
                    precision=6
                )
            )

            f.write("\n\n")

    return path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "EECE 5644 Assignment 1 Question 3 "
            "Gaussian minimum-P(error) classifiers"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("q3_outputs")
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must satisfy 0 < alpha < 1.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X_wine, y_wine, _ = load_wine(args.data_dir)

    if X_wine.shape != (4898, 11):
        raise ValueError(
            f"Unexpected Wine dataset shape: {X_wine.shape}"
        )

    wine_result = analyze_dataset(
        dataset_name="Wine Quality",
        X_raw=X_wine,
        y=y_wine,
        output_dir=args.output_dir,
        alpha=args.alpha,
        label_names=None,
    )

    X_har, y_har, _, activity_names = load_har(
        args.data_dir
    )

    if X_har.shape != (10299, 561):
        raise ValueError(
            f"Unexpected HAR dataset shape: {X_har.shape}"
        )

    har_result = analyze_dataset(
        dataset_name="Human Activity Recognition",
        X_raw=X_har,
        y=y_har,
        output_dir=args.output_dir,
        alpha=args.alpha,
        label_names=activity_names,
    )

    summary_path = write_summary(
        [wine_result, har_result],
        args.output_dir,
        args.alpha
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
