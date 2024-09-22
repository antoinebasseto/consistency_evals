import argparse

from utils import get_dataset_handler


def main(args: argparse.Namespace):
    dataset_handler = get_dataset_handler(
        args.dataset, args.engine, args.experiment, args.ai_type
    )

    dataset_handler.gen_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["mmlu", "nejm", "custom"])
    parser.add_argument(
        "--engine", "-e", choices=["gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06"]
    )
    parser.add_argument(
        "--experiment", "-ex", choices=["shuffling", "withdrawing", "control"]
    )
    parser.add_argument("--ai_type", "-ai", choices=["recommendation", "hypothesis"])

    args = parser.parse_args()
    main(args)
