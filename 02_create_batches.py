import argparse
import os

from openai import OpenAI

from config import OPENAI_API_KEY
from utils import get_dataset_handler


def main(args: argparse.Namespace):
    client = OpenAI(api_key=OPENAI_API_KEY)
    dataset_handler = get_dataset_handler(
        args.dataset, args.engine, args.experiment, args.ai_type
    )
    batch_inputs_dir = dataset_handler.get_batch_inputs_dir()

    start_file_range = max(0, args.after)
    end_file_range = min(args.after + args.n_files, len(os.listdir(batch_inputs_dir)))

    for i, input_file in enumerate(sorted(os.listdir(batch_inputs_dir))):

        if i not in range(start_file_range, end_file_range):
            continue
        print(f"Creating batch for file #{i} - {input_file}")

        input_file_path = os.path.join(batch_inputs_dir, input_file)

        batch_input_file = client.files.create(
            file=open(input_file_path, "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "dataset": args.dataset,
                "engine": args.engine,
                "experiment": args.experiment,
                "input_file": input_file,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["mmlu", "nejm", "custom"])
    parser.add_argument(
        "--engine", "-e", choices=["gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06"]
    )
    parser.add_argument("--experiment", "-ex", choices=["shuffling", "withdrawing"])
    parser.add_argument("--ai_type", "-ai", choices=["recommendation", "hypothesis"])

    parser.add_argument("--after", "-a", type=int)
    parser.add_argument("--n_files", "-n", type=int)

    args = parser.parse_args()
    main(args)
