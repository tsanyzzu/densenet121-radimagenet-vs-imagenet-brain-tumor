from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.utils.paths import format_run_id, get_configs_dir, get_run_output_dir


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Load a YAML file safely into a Python dictionary."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], file_path: Path | str) -> None:
    """Save a dictionary to a YAML file with clean formatting."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def load_full_config(configs_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load and merge dataset.yaml, model.yaml, and experiment.yaml into a unified config dict."""
    cfg_dir = Path(configs_dir) if configs_dir is not None else get_configs_dir()

    dataset_cfg = load_yaml(cfg_dir / "dataset.yaml")
    model_cfg = load_yaml(cfg_dir / "model.yaml")
    experiment_cfg = load_yaml(cfg_dir / "experiment.yaml")

    full_cfg = {
        "experiment_name": experiment_cfg.get("experiment_name", "densenet121_experiment"),
        "dataset": dataset_cfg,
        "model": model_cfg,
        "factors": experiment_cfg.get("factors", {}),
        "training": experiment_cfg.get("training", {}),
        "early_stopping": experiment_cfg.get("early_stopping", {}),
        "reproducibility": experiment_cfg.get("reproducibility", {}),
        "convergence": experiment_cfg.get("convergence", {}),
        "outputs": experiment_cfg.get("outputs", {}),
    }

    validate_protocol_constraints(full_cfg)
    return full_cfg


def validate_protocol_constraints(cfg: Dict[str, Any]) -> None:
    """Strictly assert that the configuration conforms to the mandatory research protocol.

    Raises:
        ValueError: If any mandatory parameter deviates from master.md specifications.
    """
    training = cfg.get("training", {})
    early_stopping = cfg.get("early_stopping", {})
    factors = cfg.get("factors", {})
    model = cfg.get("model", {})
    dataset = cfg.get("dataset", {})

    # 1. Optimizer constraint
    optimizer = str(training.get("optimizer", "")).lower()
    if optimizer != "adamw":
        raise ValueError(
            f"Protocol Violation: Optimizer must be 'adamw', found '{optimizer}'."
        )

    # 2. Learning rate constraints
    lr_backbone = float(training.get("lr_backbone", 0))
    if abs(lr_backbone - 1e-4) > 1e-9:
        raise ValueError(
            f"Protocol Violation: Backbone learning rate must be 1e-4, found {lr_backbone}."
        )

    lr_classifier = float(training.get("lr_classifier", 0))
    if abs(lr_classifier - 1e-3) > 1e-9:
        raise ValueError(
            f"Protocol Violation: Classifier learning rate must be 1e-3, found {lr_classifier}."
        )

    if lr_backbone == lr_classifier:
        raise ValueError(
            "Protocol Violation: Backbone and classifier must have distinct learning rates."
        )

    # 3. Batch size and epochs
    batch_size = int(training.get("batch_size", 0))
    if batch_size != 32:
        raise ValueError(
            f"Protocol Violation: Batch size must be 32, found {batch_size}."
        )

    max_epochs = int(training.get("max_epochs", 0))
    if max_epochs != 50:
        raise ValueError(
            f"Protocol Violation: Maximum epochs must be 50, found {max_epochs}."
        )

    # 4. Early stopping constraints
    if not early_stopping.get("enabled", False):
        raise ValueError("Protocol Violation: Early stopping must be enabled.")

    monitor = early_stopping.get("monitor", "")
    if monitor != "val_macro_f1":
        raise ValueError(
            f"Protocol Violation: Early stopping monitor must be 'val_macro_f1', found '{monitor}'."
        )

    mode = early_stopping.get("mode", "")
    if mode != "max":
        raise ValueError(
            f"Protocol Violation: Early stopping mode must be 'max', found '{mode}'."
        )

    patience = int(early_stopping.get("patience", 0))
    if patience != 8:
        raise ValueError(
            f"Protocol Violation: Early stopping patience must be 8, found {patience}."
        )

    min_delta = float(early_stopping.get("min_delta", 0))
    if abs(min_delta - 1e-4) > 1e-9:
        raise ValueError(
            f"Protocol Violation: Early stopping min_delta must be 1e-4, found {min_delta}."
        )

    # 5. Factorial design constraints (2 x 3 x 5 = 30 runs)
    expected_sources = ["imagenet", "radimagenet"]
    sources = factors.get("pretrained_sources", [])
    if sorted(sources) != sorted(expected_sources):
        raise ValueError(
            f"Protocol Violation: Pretrained sources must be {expected_sources}, found {sources}."
        )

    expected_strategies = ["partial_1block", "partial_2block", "full_finetuning"]
    strategies = factors.get("strategies", [])
    if sorted(strategies) != sorted(expected_strategies):
        raise ValueError(
            f"Protocol Violation: Strategies must be {expected_strategies}, found {strategies}."
        )

    expected_seeds = [42, 43, 44, 45, 46]
    seeds = factors.get("seeds", [])
    if seeds != expected_seeds:
        raise ValueError(
            f"Protocol Violation: Random seeds must be exactly {expected_seeds}, found {seeds}."
        )

    # 6. Expected trainable parameter numbers per strategy
    model_strategies = model.get("strategies", {})
    expected_param_counts = {
        "partial_1block": 2164228,
        "partial_2block": 5528324,
        "full_finetuning": 6957956,
    }
    for strat, expected_count in expected_param_counts.items():
        if strat in model_strategies:
            actual_count = model_strategies[strat].get("expected_trainable_params")
            if actual_count != expected_count:
                raise ValueError(
                    f"Protocol Violation: Strategy '{strat}' expected {expected_count} trainable params, found {actual_count}."
                )

    # 7. Dataset classes constraint (4 classes)
    class_names = dataset.get("class_names", [])
    if len(class_names) != 4:
        raise ValueError(
            f"Protocol Violation: BRISC dataset must have 4 classes, found {len(class_names)}: {class_names}."
        )


def generate_experiment_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all 30 experiment run specifications for the 2 x 3 x 5 factorial grid.

    Returns:
        List of 30 dictionary items, each uniquely specifying a run:
        - run_id: e.g. "imagenet__partial_1block__seed_42"
        - pretrained_source: "imagenet" | "radimagenet"
        - strategy: "partial_1block" | "partial_2block" | "full_finetuning"
        - seed: 42 | 43 | 44 | 45 | 46
        - expected_trainable_params: int
        - run_dir: Path to output directory
    """
    validate_protocol_constraints(cfg)

    factors = cfg["factors"]
    sources = factors["pretrained_sources"]
    strategies = factors["strategies"]
    seeds = factors["seeds"]
    model_strategies = cfg.get("model", {}).get("strategies", {})

    grid = []
    seen_run_ids = set()

    for source in sources:
        for strategy in strategies:
            strat_info = model_strategies.get(strategy, {})
            expected_params = strat_info.get("expected_trainable_params")
            freeze_bn = strat_info.get("freeze_backbone_bn_stats", False)

            for seed in seeds:
                run_id = format_run_id(source, strategy, seed)
                if run_id in seen_run_ids:
                    raise ValueError(f"Duplicate run_id generated: '{run_id}'")
                seen_run_ids.add(run_id)

                run_dir = get_run_output_dir(source, strategy, seed, create=False)

                run_spec = {
                    "run_id": run_id,
                    "pretrained_source": source,
                    "strategy": strategy,
                    "seed": seed,
                    "expected_trainable_params": expected_params,
                    "freeze_backbone_bn_stats": freeze_bn,
                    "run_dir": run_dir,
                }
                grid.append(run_spec)

    if len(grid) != 30:
        raise ValueError(
            f"Factorial design mismatch: Expected exactly 30 runs, generated {len(grid)}."
        )

    return grid


def save_config_snapshot(cfg: Dict[str, Any], output_path: Path | str) -> None:
    """Save an immutable YAML snapshot of the active configuration in a run directory."""
    save_yaml(cfg, output_path)

