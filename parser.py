from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

try:
    import yaml  # optional dependency; falls back to JSON if missing
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = ["parse_args"]


# ------------------------------------------------------------------
# Helper ----------------------------------------------------------------


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
    """Load a *YAML* or *JSON* config file into a dict."""
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed – `pip install pyyaml` or use JSON config.")
        return yaml.safe_load(text) or {}

    if path.suffix.lower() == ".json":
        return json.loads(text)

    raise ValueError("Unsupported config file type – use .yaml/.yml or .json")


# ------------------------------------------------------------------
# Public ----------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: C901 – a bit long but flat
    """Parse CLI & optional YAML/JSON config.
    ----------
    """

    parser = argparse.ArgumentParser(
        description=f"Experiment parser",
    )

    # ------------------------------------------------------------------
    # Generic experiment parameters                                     |
    # ------------------------------------------------------------------
    g = parser.add_argument_group("Problem dimensions & data generation")
    g.add_argument("--num_seeds", type=int, default=8, help="Number of random seeds per setting")
    g.add_argument("--num_measurements", type=int, default=3, help="Number of measurements")
    g.add_argument("--sequence_length", type=int, default=3, help="Length of the measurement sequence")
    g.add_argument("--teacher_rank", type=int, default=1, help="Rank of the teacher matrix")
    g.add_argument("--teacher_dim", type=int, default=2, help="Dimension of the teacher matrix")
    g.add_argument("--student_dims", type=int, default=list(range(3, 16)), help="Dimensions of the student matrix")

    g = parser.add_argument_group("Guess & Check hyperparameters")
    g.add_argument("--gnc_num_samples", type=int, default=int(1e8), help="Number of G&C samples")
    g.add_argument("--gnc_batch_size", type=int, default=int(1e7), help="Batch sizes for G&C")
    g.add_argument("--gnc_eps_train", type=float, default=0.001, help="Training loss threshold for successful G&C trial")

    g = parser.add_argument_group("Gradient Descent hyper‑parameters")
    g.add_argument("--gd_lr", type=float, default=1e-2, help="Learning rate for GD")
    g.add_argument("--gd_epochs", type=int, default=int(1e5), help="Number of epochs for GD")
    g.add_argument("--gd_init_scale", type=float, default=1e-2, help="Initialization scale for GD")

    # ------------------------------------------------------------------
    # Meta                                                               |
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Optional YAML or JSON config file (CLI flags override)",
    )
    parser.add_argument("--results_dir", type=pathlib.Path, default=pathlib.Path("./results"), help="Results directory")
    parser.add_argument("--figures_dir", type=pathlib.Path, default=pathlib.Path("./figures"), help="Figures directory")

    # ---- first round parse (just to grab --config) --------------------
    if "--config" in parser.parse_known_args()[0]._get_args():  # type: ignore
        # weird, but keeps mypy happy; we only want to sniff existence
        pass

    args, unknown = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Config file overrides                                              |
    # ------------------------------------------------------------------
    if args.config is not None:
        cfg_dict = _load_config(args.config)
        parser.set_defaults(**cfg_dict)
        # re‑parse now with new defaults (and original CLI flags win)
        args = parser.parse_args()

    # Final tweaks ------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    return args