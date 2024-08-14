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
    SYSTEM_PROMPT = {
        "recommendation": "You are a helpful assistant.",
        "hypothesis": "",
    }
    MESSAGE_PROMPT_START = {
        "recommendation": """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
""".strip(),
        "hypothesis": "",
    }
    ANSWER_PATTERN_MULTICHOICE = r""

    def __init__(self, engine: str, experiment: str, ai_type: str) -> None:
        if ai_type == "hypothesis":
            raise NotImplementedError("hypothesis-driven AI is not supported for MMLU")
        super().__init__("mmlu", engine, experiment, ai_type)

    def _load_dataframes(self, subject: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dev_df = pd.read_csv(
            os.path.join(self.get_raw_data_dir(), "dev", f"{subject}_dev.csv"),
            header=None,
            names=["question", "A", "B", "C", "D", "answer"],
        )
        test_df = pd.read_csv(
            os.path.join(self.get_raw_data_dir(), "test", f"{subject}_test.csv"),
            header=None,
            names=["question", "A", "B", "C", "D", "answer"],
        )
        return dev_df, test_df

    def _format_subject(self, subject: str) -> str:
        return " ".join(subject.split("_"))

    def gen_batch_inputs(
        self, k_shot: int = 0, max_options_lists: int = -1, temperature: float = 0.0
    ) -> None:
        if k_shot != 0:
            raise NotImplementedError("k-shot learning is not supported for MMLU")

        for subject in subjects:
            _, test_df = self._load_dataframes(subject)
            input_data = []
            for i in range(test_df.shape[0]):
                original_options = test_df[["A", "B", "C", "D"]].iloc[i].tolist()
                options_lists = self._gen_experiment_options_lists(
                    original_options, max_options_lists
                )

                for options in options_lists:
                    prompt = self._gen_prompt(str(test_df.iloc[i, 0]), options)
                    input_data.append(
                        self._gen_batch_input_line(
                            f"{subject}_{i}_{options}",
                            prompt,
                            temperature=temperature,
                            force_json=False,
                        )
                    )

            with open(
                os.path.join(self.get_batch_inputs_dir(), f"input_{subject}.jsonl"),
                "w",
            ) as file:
                for entry in input_data:
                    json_line = json.dumps(entry)
                    file.write(json_line + "\n")
