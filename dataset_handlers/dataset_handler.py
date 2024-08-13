import itertools
import os
import random
import string
from typing import List, Optional, Tuple


class DatasetHandler:
    SYSTEM_PROMPT = {}
    MESSAGE_PROMPT_START = {}
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"

    def __init__(
        self, dataset_name: str, engine: str, experiment: str, ai_type: str
    ) -> None:
        self.dataset_name = dataset_name
        self.engine = engine
        self.experiment = experiment
        self.ai_type = ai_type

        dirs = [
            self.get_raw_data_dir(),
            self.get_batch_inputs_dir(),
            self.get_batch_outputs_dir(),
            self.get_parsed_batch_outputs_dir(),
            self.get_results_dir(),
        ]

        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def get_raw_data_dir(self) -> str:
        return os.path.join("data", self.dataset_name, "01_raw")

    def get_batch_inputs_dir(self) -> str:
        return os.path.join(
            "data",
            self.dataset_name,
            "02_batch_inputs",
            self.engine,
            self.ai_type,
            self.experiment,
        )

    def get_batch_outputs_dir(self) -> str:
        return os.path.join(
            "data",
            self.dataset_name,
            "03_batch_outputs",
            self.engine,
            self.ai_type,
            self.experiment,
        )

    def get_parsed_batch_outputs_dir(self) -> str:
        return os.path.join(
            "data",
            self.dataset_name,
            "04_parsed_batch_outputs",
            self.engine,
            self.ai_type,
            self.experiment,
        )

    def get_results_dir(self) -> str:
        return os.path.join(
            "data",
            self.dataset_name,
            "05_results",
            self.engine,
            self.ai_type,
            self.experiment,
        )

    def _gen_experiment_options_lists(
        self, option: List[str], max_options_lists: int = -1
    ) -> List[Tuple[str, ...]]:
        if self.experiment == "shuffling":
            options_lists = list(itertools.permutations(option))
            if max_options_lists != -1:
                max_options_lists = min(max_options_lists, len(options_lists))
                options_lists = random.sample(options_lists, max_options_lists)
            return options_lists
        if self.experiment == "withdrawing":
            options_lists = list(itertools.combinations(option, len(option) - 1))
            if max_options_lists != -1:
                max_options_lists = min(max_options_lists, len(options_lists))
                options_lists = random.sample(options_lists, max_options_lists)
            return options_lists
        else:
            raise ValueError(f"Unknown experiment: {self.experiment}")

    def _gen_batch_input_line(
        self,
        custom_id: str,
        prompt: str,
        temperature: int = 0,
        force_json: bool = False,
        json_schema: Optional[object] = None,
    ) -> dict:
        assert (force_json and json_schema) or (not force_json and json_schema is None)
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.engine,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT[self.ai_type]},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "response_format": (
                    {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    }
                    if force_json
                    else {}
                ),
            },
        }

    def _gen_prompt(self, question: str, options: Tuple[str, ...]) -> str:
        prompt = f"{self.MESSAGE_PROMPT_START[self.ai_type]}\n\n{question}\n\n"
        for i, option in enumerate(options):
            prompt += f"{string.ascii_uppercase[i]}. {option}\n"
        return prompt
