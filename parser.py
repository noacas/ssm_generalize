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
    g.add_argument("--sequence_length", type=int, default=5, help="Length of the measurement sequence")
    g.add_argument("--student_dims", type=int, nargs='+', default=list(range(100, 300, 10)), help="Student dimensions (one or more integers)")
    g.add_argument("--eps_train", type=float, default=float(1e-3), help="Training loss threshold for successful trial")

    g = parser.add_argument_group("Guess & Check hyperparameters")
    g.add_argument(
        "--gnc", dest="gnc", action="store_true", help="Enable Guess & Check (default: True)", default=True
    )
    g.add_argument(
        "--no-gnc", dest="gnc", action="store_false", help="Disable Guess & Check"
    )
    g.add_argument("--gnc_num_samples", type=int, default=int(1e10), help="Number of G&C samples")
    g.add_argument("--gnc_batch_size", type=int, default=int(1e6), help="Batch sizes for G&C")

    g = parser.add_argument_group("Gradient Descent hyper‑parameters")
    g.add_argument(
        "--gd", dest="gd", action="store_true", help="Enable Gradient Descent", default=False
    )
    g.add_argument(
        "--no-gd", dest="gd", action="store_false", help="Disable Gradient Descent"
    )
    g.add_argument("--gd_lr", type=float, default=1e-3, help="Learning rate for GD")
    g.add_argument("--gd_epochs", type=int, default=int(1e6), help="Number of epochs for GD")
    g.add_argument("--gd_init_scale", type=float, default=1e-2, help="Initialization scale for GD")
    g.add_argument("--gd_optimizer", type=str, default="adam", help="Optimizer for GD", choices=["adam", "gd"])

    # ------------------------------------------------------------------
    # Meta                                                               |
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Optional YAML or JSON config file (CLI flags override)",
    )
    parser.add_argument("--results_dir", type=pathlib.Path, default=pathlib.Path("./test_results/results"), help="Results directory")
    parser.add_argument("--figures_dir", type=pathlib.Path, default=pathlib.Path("./test_results/figures"), help="Figures directory")
    parser.add_argument("--checkpoint_dir", type=pathlib.Path, default=pathlib.Path("./test_results/checkpoints"), help="Checkpoints directory")
    parser.add_argument("--checkpoint_interval", type=int, default=3600, help="Checkpoint interval in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from the latest checkpoint if available")
    parser.add_argument("--log_dir", type=pathlib.Path, default=pathlib.Path("./test_results/logs"), help="Logs directory")

    g = parser.add_argument_group("GPU settings")
    g.add_argument("--max_gpus", type=int, default=4, help="Maximum number of GPUs to use")

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
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to lists in case config provided single ints
    if not isinstance(args.student_dims, (list, tuple)):
        args.student_dims = [int(args.student_dims)]

    return args