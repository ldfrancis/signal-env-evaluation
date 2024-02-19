import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()