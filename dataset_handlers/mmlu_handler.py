import itertools
import json
import os
import random
from typing import List, Tuple

import pandas as pd

from dataset_handlers.dataset_handler import DatasetHandler

subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

choices = ["A", "B", "C", "D"]


class MMLUHandler(DatasetHandler):
    SYSTEM_PROMPT = "You are a helpful assistant."
    MESSAGE_PROMPT_START = ""
    ANSWER_PATTERN_MULTICHOICE = r""

    def __init__(self, engine: str, experiment: str) -> None:
        super().__init__("mmlu", engine, experiment)

    def _load_dataframes(self, subject: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dev_df = pd.read_csv(
            os.path.join(self.get_raw_data_dir(), "dev", f"{subject}_dev.csv"),
            header=None,
        )
        test_df = pd.read_csv(
            os.path.join(self.get_raw_data_dir(), "test", f"{subject}_test.csv"),
            header=None,
        )
        return dev_df, test_df

    def _format_subject(self, subject: str) -> str:
        return " ".join(subject.split("_"))

    def _format_example(
        self,
        df: pd.DataFrame,
        idx: int,
        option: Tuple[str, ...] = ("A", "B", "C", "D"),
        include_answer: bool = True,
    ) -> str:
        """
        Format a single example from the DataFrame into a prompt string.
        """
        prompt = df.iloc[idx, 0]
        num_options = len(option)
        option_indices = [choices.index(option) for option in option]

        for j, option_index in enumerate(option_indices):
            prompt += f"\n{choices[j]}. {df.iloc[idx, option_index + 1]}"

        prompt += "\nAnswer:"
        if include_answer:
            prompt += f" {df.iloc[idx, num_options + 1]}\n\n"

        return prompt

    def _gen_prompt(self, train_df: pd.DataFrame, subject: str, k_shot: int) -> str:
        prompt = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}.\n\n"
        for i in range(k_shot):
            prompt += self._format_example(train_df, i)
        return prompt

    def gen_batch_inputs(self, k_shot: int, max_options_lists: int = -1) -> None:
        if k_shot != 0:
            raise NotImplementedError("k-shot learning is not supported yet for MMLU")

        for subject in subjects:
            dev_df, test_df = self._load_dataframes(subject)

            input_data = []
            for i in range(test_df.shape[0]):

                train_prompt = self._gen_prompt(dev_df, subject, k_shot)

                original_options = ["A", "B", "C", "D"]
                options_lists = self._gen_experiment_options_lists(
                    original_options, max_options_lists
                )

                for options in options_lists:

                    input_data.append(
                        self._gen_batch_input_line(
                            f"{subject}_{question_id}_{options}", prompt
                        )
                    )

            with open(
                os.path.join(self.get_batch_inputs_dir(), f"input_{subject}.jsonl"),
                "w",
            ) as file:
                for entry in input_data:
                    json_line = json.dumps(entry)
                    file.write(json_line + "\n")
