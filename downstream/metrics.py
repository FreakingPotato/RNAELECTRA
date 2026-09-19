"""
Metrics Module for BEACON Benchmark
Contains metric computation functions aligned with BEACON Paper Table 1.
Covers the function and engineering tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr, pearsonr

def compute_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute accuracy for classification tasks."""
    preds = np.argmax(predictions, axis=1)
    return float(accuracy_score(labels, preds))


def compute_f1_macro(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute macro F1 score."""
    preds = np.argmax(predictions, axis=1)
    return float(f1_score(labels, preds, average='macro', zero_division=0))


def compute_f1_multilabel(labels: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> float:
    """Compute F1 score for multi-label classification."""
    preds_binary = (predictions > threshold).astype(int)
    return float(f1_score(labels, preds_binary, average='macro', zero_division=0))


def compute_mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute Matthews Correlation Coefficient."""
    preds = np.argmax(predictions, axis=1)
    return float(matthews_corrcoef(labels, preds))


def compute_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute AUC score."""
    try:
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            return float(roc_auc_score(labels, predictions.flatten()))
        elif len(np.unique(labels)) == 2:
            return float(roc_auc_score(labels, predictions[:, 1]))
        else:
            return float(roc_auc_score(labels, predictions, multi_class='ovr'))
    except Exception:
        return 0.0


def compute_auc_multilabel(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute AUC for multi-label classification."""
    try:
        return float(roc_auc_score(labels, predictions, average='macro'))
    except Exception:
        return 0.0


def compute_r2(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute R2 score for regression."""
    preds = predictions.squeeze()
    labels_flat = labels.squeeze() if labels.ndim > 1 else labels
    return float(r2_score(labels_flat.flatten(), preds.flatten()))


def compute_pearson_r2_multi(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """
    Compute Pearson correlation squared for multi-output regression (BEACON style for PRS).
    """
    results = {}
    num_outputs = labels.shape[1] if labels.ndim > 1 else 1

    r2_values = []
    for i in range(num_outputs):
        corr = pearsonr(labels[:, i], predictions[:, i])[0] ** 2
        r2_values.append(corr)

    if num_outputs == 3:  # PRS task
        results["r2_ON"] = r2_values[0]
        results["r2_OFF"] = r2_values[1]
        results["r2_ON_OFF"] = r2_values[2]

    results["r2"] = float(np.mean(r2_values))
    return results


def compute_spearman(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute Spearman correlation coefficient."""
    preds = predictions.squeeze()
    labels_flat = labels.squeeze() if labels.ndim > 1 else labels
    corr, _ = spearmanr(labels_flat.flatten(), preds.flatten())
    return float(corr) if not np.isnan(corr) else 0.0


def compute_top_k_accuracy_spliceai(predictions: np.ndarray, labels: np.ndarray, class_index: int) -> float:
    """
    Adopted from compute_top_k_accuracy_spliceai in BEACON codebase, tailored for splice site prediction.
    Calculate top-k accuracy for a specific class (BEACON style).

    Args:
        predictions: shape (num_samples, seq_length, num_classes) - logits/scores
        labels: shape (num_samples, seq_length) - class labels (0, 1, or 2)
        class_index: which class to evaluate (1=acceptor, 2=donor)

    Returns:
        top_k_accuracy for the specified class
    """
    # Get scores for the specified class and flatten
    class_scores = predictions[:, :, class_index].flatten()

    # Convert labels to one-hot and get the column for this class
    labels_flat = labels.flatten().astype(int)
    one_hot = np.zeros((labels_flat.size, 3))
    one_hot[np.arange(labels_flat.size), labels_flat] = 1
    class_labels = one_hot[:, class_index]

    # k = number of true positives for this class
    k = int(np.sum(class_labels))

    if k == 0:
        return 0.0

    # Get indices of top-k highest scores
    top_k_indices = np.argsort(class_scores)[::-1][:k]

    # Count how many of top-k are true positives
    true_positives = np.sum(class_labels[top_k_indices])

    return float(true_positives / k)


def compute_metrics_fn(task_cfg: dict):
    """
    Return a compute_metrics function for the HuggingFace Trainer based on task config.

    Args:
        task_cfg: Task configuration dictionary containing 'metric' and 'problem_type'

    Returns:
        A function that computes metrics given (predictions, labels)
    """
    metric_name = task_cfg['metric']
    problem_type = task_cfg['problem_type']

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        results = {}

        if problem_type == "single_label_classification":
            preds = np.argmax(predictions, axis=1)

            # Primary metric
            if metric_name == "accuracy":
                results["accuracy"] = compute_accuracy(labels, predictions)
            elif metric_name == "top_k_accuracy":
                results["top_k_accuracy"] = compute_accuracy(labels, predictions)
            elif metric_name == "f1":
                results["f1"] = compute_f1_macro(labels, predictions)

            # Additional metrics
            results["f1_macro"] = compute_f1_macro(labels, predictions)
            results["mcc"] = compute_mcc(labels, predictions)
            results["auc"] = compute_auc(labels, predictions)

        elif problem_type == "multi_label_classification":
            # Primary metric
            if metric_name == "f1":
                results["f1"] = compute_f1_multilabel(labels, predictions)
            elif metric_name == "auc":
                results["auc"] = compute_auc_multilabel(labels, predictions)

        elif problem_type == "token_classification":
            if metric_name == "top_k_accuracy":
                # BEACON-style top-k accuracy for splice site prediction
                # predictions shape: (batch, seq_len, num_classes)
                # labels shape: (batch, seq_len)

                acceptor_acc = compute_top_k_accuracy_spliceai(predictions, labels, class_index=1)
                donor_acc = compute_top_k_accuracy_spliceai(predictions, labels, class_index=2)

                results["acceptor_topk_acc"] = acceptor_acc
                results["donor_topk_acc"] = donor_acc
                results["top_k_accuracy"] = (acceptor_acc + donor_acc) / 2  # avg of acceptor and donor

            # Additional metrics for debugging
            preds_flat = np.argmax(predictions, axis=-1).flatten()
            labels_flat = labels.flatten()
            mask = labels_flat != -100
            results["accuracy_all"] = float(accuracy_score(labels_flat[mask], preds_flat[mask]))

        else:  # regression
            preds = predictions.squeeze()
            labels_flat = labels.squeeze() if labels.ndim > 1 else labels

            # Multi-output regression with Pearson R² (for PRS task)
            if preds.ndim > 1 and preds.shape[-1] > 1 and metric_name == "r2":
                # Use Pearson R² per column
                results.update(compute_pearson_r2_multi(labels_flat, preds))
            else:
                # Single-output regression
                preds_for_metric = preds
                labels_for_metric = labels_flat

                if metric_name == "r2":
                    results["r2"] = compute_r2(labels_for_metric, preds_for_metric)
                elif metric_name == "spearman":
                    results["spearman"] = compute_spearman(labels_for_metric, preds_for_metric)

            # Additional metrics for regression (if not already computed)
            if "r2" not in results:
                if preds.ndim > 1 and preds.shape[-1] > 1:
                    preds_for_r2 = preds.mean(axis=-1)
                    labels_for_r2 = labels_flat.mean(axis=-1) if labels_flat.ndim > 1 else labels_flat
                else:
                    preds_for_r2 = preds
                    labels_for_r2 = labels_flat
                results["r2"] = compute_r2(labels_for_r2, preds_for_r2)
            if "spearman" not in results:
                if preds.ndim > 1 and preds.shape[-1] > 1:
                    preds_for_sp = preds.mean(axis=-1)
                    labels_for_sp = labels_flat.mean(axis=-1) if labels_flat.ndim > 1 else labels_flat
                else:
                    preds_for_sp = preds
                    labels_for_sp = labels_flat
                results["spearman"] = compute_spearman(labels_for_sp, preds_for_sp)

        return results

    return compute_metrics