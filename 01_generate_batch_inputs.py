import argparse

from utils import get_dataset_handler


def main(args: argparse.Namespace):
    dataset_handler = get_dataset_handler(
        args.dataset, args.engine, args.experiment, args.ai_type
    )

    dataset_handler.gen_batch_inputs(args.k_shot, args.max_options_lists, args.temperature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["mmlu", "nejm", "custom"])
    parser.add_argument(
        "--engine", "-e", choices=["gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06"]
    )
    parser.add_argument("--experiment", "-ex", choices=["shuffling", "withdrawing"])
    parser.add_argument("--ai_type", "-ai", choices=["recommendation", "hypothesis"])
    parser.add_argument("--temperature", "-t", type=float, default=0.0)

    parser.add_argument("--k_shot", "-k", type=int, default=0)
    parser.add_argument("--max_options_lists", "-m", type=int, default=-1)

    args = parser.parse_args()
    main(args)
