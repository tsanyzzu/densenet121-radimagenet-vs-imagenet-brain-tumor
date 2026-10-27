from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Resolves by locating the parent of the `src/` directory where this file resides.
    """
    # This file is located at <project_root>/src/utils/paths.py
    current_file = Path(__file__).resolve()
    # Go up 3 levels: paths.py -> utils -> src -> <project_root>
    root = current_file.parents[2]
    return root


def get_configs_dir() -> Path:
    """Return path to configs/ directory."""
    return get_project_root() / "configs"


def get_data_dir() -> Path:
    """Return path to data/ directory."""
    return get_project_root() / "data"


def get_splits_dir() -> Path:
    """Return path to data/splits/ directory, creating it if it doesn't exist."""
    splits_dir = get_data_dir() / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    return splits_dir


def get_outputs_dir() -> Path:
    """Return path to outputs/ directory, creating it if it doesn't exist."""
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def format_run_id(pretrained_source: str, strategy: str, seed: int) -> str:
    """Generate standard unique run ID following the research protocol.

    Example:
        format_run_id("imagenet", "partial_1block", 42)
        -> "imagenet__partial_1block__seed_42"
    """
    return f"{pretrained_source}__{strategy}__seed_{seed}"


def parse_run_id(run_id: str) -> dict[str, str | int]:
    """Parse a run ID into its constituent experiment factors.

    Example:
        parse_run_id("imagenet__partial_1block__seed_42")
        -> {"pretrained_source": "imagenet", "strategy": "partial_1block", "seed": 42}
    """
    parts = run_id.split("__")
    if len(parts) != 3 or not parts[2].startswith("seed_"):
        raise ValueError(
            f"Invalid run_id format: '{run_id}'. Expected '<source>__<strategy>__seed_<int>'."
        )
    return {
        "pretrained_source": parts[0],
        "strategy": parts[1],
        "seed": int(parts[2].replace("seed_", "")),
    }


def get_run_output_dir(
    pretrained_source: str,
    strategy: str,
    seed: int,
    base_output_dir: Optional[Path] = None,
    create: bool = True,
) -> Path:
    """Return the output directory path for a specific experiment run.

    Structure:
        <base_output_dir>/runs/<pretrained_source>/<strategy>/seed_<seed>/

    Args:
        pretrained_source: 'imagenet' or 'radimagenet'
        strategy: 'partial_1block', 'partial_2block', or 'full_finetuning'
        seed: Random seed integer (e.g. 42)
        base_output_dir: Base directory (defaults to <project_root>/outputs)
        create: Whether to create the directory on disk if missing.
    """
    base = base_output_dir if base_output_dir is not None else get_outputs_dir()
    run_dir = base / "runs" / pretrained_source / strategy / f"seed_{seed}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_posix_str(path: Path | str) -> str:
    """Convert any filesystem path to a forward-slash POSIX string for manifest portability."""
    return Path(path).as_posix()

