import argparse
import yaml
from pathlib import Path

from classes.pipeline import RadiomicsPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract radiomics features from PROSTATEx data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    pipeline = RadiomicsPipeline(config)
    pipeline.save_config_snapshot()
    manifest_path = Path(config["manifest_csv"])
    pipeline.build_dataset(manifest_path)


if __name__ == "__main__":
    main()

