import os
import sys
import subprocess
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main orchestrator that launches train.py as subprocess for a single run.

    Args:
        cfg: Hydra configuration object
    """
    # Support both 'run' and 'run_id' parameters
    if "run" in cfg and cfg.run is not None and cfg.run_id is None:
        cfg.run_id = cfg.run

    # Validate required parameters
    if "run_id" not in cfg or cfg.run_id is None:
        raise ValueError("run_id must be specified via CLI: run_id={run_id}")
    
    if "mode" not in cfg or cfg.mode not in ["trial", "full"]:
        raise ValueError(f"mode must be 'trial' or 'full', got {cfg.get('mode', 'MISSING')}")
    
    run_id = cfg.run_id
    mode = cfg.mode
    results_dir = cfg.get("results_dir", "results")
    
    # Get results directory from config or CLI
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting experiment: run_id={run_id}, mode={mode}")
    logger.info(f"Results directory: {results_dir_path}")
    
    # Load run-specific configuration to validate it exists
    run_config_path = Path("config") / "runs" / f"{run_id}.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    
    # Build command with all necessary parameters
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run_id={run_id}",
        f"results_dir={results_dir}",
        f"mode={mode}",
    ]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        logger.error(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
