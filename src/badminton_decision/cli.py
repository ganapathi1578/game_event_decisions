def download_main():
    from badminton_decision.data.download import main
    main()


def prepare_main():
    from badminton_decision.data.prepare import main
    main()


def train_main():
    import argparse
    from badminton_decision.training.trainer import train
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name", default="badminton-jev-smolvlm2")
    args = parser.parse_args()
    train(args.config, args.name)


def evaluate_main():
    from badminton_decision.evaluation.evaluate import main
    main()


def predict_main():
    from badminton_decision.inference import main
    main()


def inspect_main():
    import argparse
    import json
    from pathlib import Path
    from badminton_decision.data.annotations import parse_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/finebadminton-20k")
    args = parser.parse_args()
    _, summary = parse_dataset(args.data_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
