"""Dependency-light linear probes and metrics."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _macro_f1(prediction: Tensor, target: Tensor, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        pred = prediction == class_id
        true = target == class_id
        tp = (pred & true).sum().item()
        fp = (pred & ~true).sum().item()
        fn = (~pred & true).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / max(1, len(scores)))


def _multiclass_metrics(logits: Tensor, target: Tensor, num_classes: int) -> dict[str, float]:
    prediction = logits.argmax(dim=-1)
    return {
        "accuracy": float((prediction == target).float().mean().item()),
        "macro_f1": _macro_f1(prediction, target, num_classes),
    }


def _binary_f1(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


def _multilabel_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    prediction = logits >= 0
    target = target.bool()
    per_label = [_binary_f1(prediction[:, i], target[:, i]) for i in range(target.size(1))]
    return {
        "micro_f1": _binary_f1(prediction, target),
        "macro_f1": float(sum(per_label) / max(1, len(per_label))),
        "subset_accuracy": float((prediction == target).all(dim=1).float().mean().item()),
    }


def _regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    error = prediction - target
    return {
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
    }


def _standardize(train: Tensor, *others: Tensor) -> tuple[Tensor, ...]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple((features - mean) / std for features in (train, *others))


def train_linear_probe(
    train_features: Tensor,
    train_targets: Tensor,
    eval_sets: dict[str, tuple[Tensor, Tensor]],
    *,
    task: str,
    output_dim: int,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "auto",
    patience: int = 10,
    min_delta: float = 1e-4,
    validation_split: float = 0.1,
    seed: int = 0,
    training_info: dict[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit a frozen-feature linear head and return metrics for each split.

    ``task`` is one of ``multiclass``, ``multilabel``, ``binary``, or
    ``regression``. Standardization statistics are computed from the fitting
    split only. Early stopping monitors the official ``valid`` split when it
    exists; otherwise a deterministic fraction of ``train`` is held out.
    """

    if train_features.ndim != 2:
        raise ValueError("train_features must have shape [N, D].")
    if train_features.size(0) == 0:
        raise ValueError("The training split is empty.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1).")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_names = list(eval_sets)
    eval_targets = [targets for _, targets in eval_sets.values()]

    fit_features = train_features
    fit_targets = train_targets
    validation_features_raw: Tensor | None = None
    validation_targets: Tensor | None = None
    if "valid" in eval_sets:
        validation_source = "valid"
        validation_features_raw, validation_targets = eval_sets["valid"]
    elif validation_split > 0.0 and train_features.size(0) >= 2:
        validation_source = "internal_train_split"
        split_generator = torch.Generator(device="cpu")
        split_generator.manual_seed(seed)
        validation_size = max(1, int(round(train_features.size(0) * validation_split)))
        validation_size = min(validation_size, train_features.size(0) - 1)
        permutation = torch.randperm(train_features.size(0), generator=split_generator)
        validation_indices = permutation[:validation_size]
        fit_indices = permutation[validation_size:]
        validation_features_raw = train_features[validation_indices]
        validation_targets = train_targets[validation_indices]
        fit_features = train_features[fit_indices]
        fit_targets = train_targets[fit_indices]
    else:
        validation_source = "none"

    standardized = _standardize(
        fit_features.float(),
        *(features.float() for features, _ in eval_sets.values()),
        *(  # Keep internal validation separate from reported evaluation splits.
            [validation_features_raw.float()]
            if validation_source == "internal_train_split" and validation_features_raw is not None
            else []
        ),
    )
    fit_features = standardized[0]
    if validation_source == "internal_train_split":
        eval_features = list(standardized[1:-1])
        validation_features = standardized[-1]
    else:
        eval_features = list(standardized[1:])
        validation_features = eval_features[eval_names.index("valid")] if validation_source == "valid" else None

    train_features = fit_features
    train_targets = fit_targets
    validation_loss_targets: Tensor | None = validation_targets

    if task == "multiclass":
        train_targets = train_targets.long()
        if validation_loss_targets is not None:
            validation_loss_targets = validation_loss_targets.long()
        head = nn.Linear(train_features.size(1), output_dim)
        counts = torch.bincount(train_targets, minlength=output_dim).float()
        class_weight = (counts.sum() / counts.clamp_min(1.0)).clamp(max=10.0)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
    elif task in {"binary", "multilabel"}:
        train_targets = train_targets.float()
        if task == "binary":
            train_targets = train_targets.reshape(-1, 1)
        if validation_loss_targets is not None:
            validation_loss_targets = validation_loss_targets.float()
            if task == "binary":
                validation_loss_targets = validation_loss_targets.reshape(-1, 1)
        head = nn.Linear(train_features.size(1), output_dim)
        if task == "binary":
            positive = train_targets.sum().clamp_min(1.0)
            negative = train_targets.numel() - positive
            pos_weight = (negative / positive).clamp(max=100.0).reshape(1).to(device)
        else:
            positive = train_targets.sum(dim=0).clamp_min(1.0)
            negative = train_targets.size(0) - positive
            pos_weight = (negative / positive).clamp(max=100.0).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "regression":
        train_targets = train_targets.float()
        if validation_loss_targets is not None:
            validation_loss_targets = validation_loss_targets.float()
        head = nn.Linear(train_features.size(1), output_dim)
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown probe task: {task}")

    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_features = train_features.contiguous()
    train_targets = train_targets.contiguous()

    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(seed + 1)
    validation_features_device = validation_features.to(device) if validation_features is not None else None
    validation_targets_device = validation_loss_targets.to(device) if validation_loss_targets is not None else None
    patience = max(1, int(patience))
    min_delta = max(0.0, float(min_delta))
    best_state: dict[str, Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(epochs):
        head.train()
        order = torch.randperm(train_features.size(0), generator=shuffle_generator)
        for start in range(0, train_features.size(0), batch_size):
            indices = order[start : start + batch_size]
            logits = head(train_features[indices].to(device))
            loss = loss_fn(logits, train_targets[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1

        if validation_features_device is not None and validation_targets_device is not None:
            head.eval()
            with torch.no_grad():
                validation_logits = head(validation_features_device)
                validation_loss = float(loss_fn(validation_logits, validation_targets_device).item())
            if validation_loss < best_validation_loss - min_delta:
                best_validation_loss = validation_loss
                best_epoch = epochs_run
                epochs_without_improvement = 0
                best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        else:
            best_epoch = epochs_run

    if best_state is not None:
        head.load_state_dict(best_state)

    if training_info is not None:
        training_info.update(
            {
                "max_epochs": epochs,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "early_stopped": epochs_run < epochs,
                "patience": patience,
                "min_delta": min_delta,
                "validation_source": validation_source,
                "validation_size": int(validation_features.size(0)) if validation_features is not None else 0,
                "fit_train_size": int(train_features.size(0)),
                "best_validation_loss": best_validation_loss if best_state is not None else None,
            }
        )

    results: dict[str, dict[str, float]] = {}
    head.eval()
    with torch.no_grad():
        for name, features, targets in zip(eval_names, eval_features, eval_targets, strict=True):
            prediction = head(features.to(device)).cpu()
            targets = targets.cpu()
            if task == "multiclass":
                results[name] = _multiclass_metrics(prediction, targets.long(), output_dim)
            elif task == "binary":
                target = targets.float().reshape(-1)
                results[name] = {
                    "f1": _binary_f1(prediction.reshape(-1) >= 0, target >= 0.5),
                    "accuracy": float(((prediction.reshape(-1) >= 0) == (target >= 0.5)).float().mean().item()),
                }
            elif task == "multilabel":
                results[name] = _multilabel_metrics(prediction, targets.float())
            else:
                results[name] = _regression_metrics(prediction, targets.float())
    return results
