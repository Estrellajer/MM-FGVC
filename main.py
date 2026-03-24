import hydra
from omegaconf import DictConfig, OmegaConf

from src.runner import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run_experiment(cfg)
    print(OmegaConf.to_yaml({"metrics": result["metrics"], "saved": result["saved_paths"]}))


if __name__ == "__main__":
    main()
