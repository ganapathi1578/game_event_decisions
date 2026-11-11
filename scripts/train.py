import argparse
from badminton_decision.training.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name", default="badminton-jev-smolvlm2")
    args = parser.parse_args()
    train(args.config, args.name)
