from __future__ import annotations

from collections import defaultdict
import math


def classification_metrics(y_true: list[str], y_pred: list[str], confidences: list[float] | None = None, bins: int = 15):
    if not y_true:
        return {"count": 0, "accuracy": 0.0, "macro_f1": 0.0}
    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    f1_values = []
    per_class = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[label] = {"support": sum(x == label for x in y_true), "precision": precision, "recall": recall, "f1": f1}
    ece = None
    if confidences is not None and len(confidences) == len(y_true):
        ece = 0.0
        for idx in range(bins):
            lo, hi = idx / bins, (idx + 1) / bins
            selected = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (idx == bins - 1 and c == 1.0)]
            if selected:
                accuracy = sum(y_true[i] == y_pred[i] for i in selected) / len(selected)
                avg_conf = sum(confidences[i] for i in selected) / len(selected)
                ece += len(selected) / len(y_true) * abs(accuracy - avg_conf)
    return {
        "count": len(y_true),
        "accuracy": correct / len(y_true),
        "macro_f1": sum(f1_values) / len(f1_values),
        "ece": ece,
        "per_class": per_class,
    }
