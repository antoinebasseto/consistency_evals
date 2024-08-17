import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from utils import get_dataset_handler


def main(args: argparse.Namespace):
    dataset_handler = get_dataset_handler(
        args.dataset, args.engine, args.experiment, args.ai_type
    )
    batch_outputs_dir = dataset_handler.get_batch_outputs_dir()
    parsed_outputs_dir = dataset_handler.get_parsed_batch_outputs_dir()
    for output_file_name in tqdm(os.listdir(batch_outputs_dir)):
        output_file_path = os.path.join(batch_outputs_dir, output_file_name)
        parsed_file_name = ""
        batch_output = []
        with open(output_file_path, "r") as output_file:
            for i, line in enumerate(output_file):
                if i == 0:
                    parsed_file_name = "_".join(
                        json.loads(line)["custom_id"].split("_")[:-1]
                    )
                data = dataset_handler.parse_batch_output_line(line)
                batch_output.extend(data)

        parsed_file_name += f"_{output_file_name.removesuffix('.jsonl')}.csv"
        pd.DataFrame(batch_output).to_csv(
            os.path.join(parsed_outputs_dir, parsed_file_name), index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["mmlu", "nejm", "custom"])
    parser.add_argument(
        "--engine", "-e", choices=["gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06"]
    )
    parser.add_argument("--experiment", "-ex", choices=["shuffling", "withdrawing"])
    parser.add_argument("--ai_type", "-ai", choices=["recommendation", "hypothesis"])

    args = parser.parse_args()
    main(args)
