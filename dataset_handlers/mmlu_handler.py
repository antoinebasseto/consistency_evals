import ast
import json
import os
import re
import string
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from tqdm import tqdm

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
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"

    def __init__(self, engine: str, experiment: str, ai_type: str) -> None:
        if ai_type == "hypothesis":
            raise NotImplementedError("hypothesis-driven AI is not supported for MMLU")
        super().__init__("mmlu", engine, experiment, ai_type)

    def gen_batch_inputs(
        self, k_shot: int = 0, max_options_lists: int = -1, temperature: float = 0.0
    ) -> None:
        if k_shot != 0:
            raise NotImplementedError("k-shot learning is not supported for MMLU")

        for subject in tqdm(subjects):
            _, test_df = self._load_dataframes(subject)
            input_data = []

            for i in range(test_df.shape[0]):
                original_options = ["A", "B", "C", "D"]
                options_lists = self._gen_experiment_options_lists(
                    original_options, max_options_lists
                )
                for options in options_lists:
                    prompt = self._gen_prompt(
                        str(test_df.iloc[i, 0]),
                        test_df[list(options)].iloc[i].to_list(),
                    )
                    input_data.append(
                        self._gen_batch_input_line(
                            f"{subject}_{i}_{options}",
                            prompt,
                            temperature=temperature,
                            force_json=False,
                        )
                    )

            with open(
                os.path.join(
                    self.get_batch_inputs_dir(),
                    f"input_{subject}_temp_{temperature}.jsonl",
                ),
                "w",
            ) as file:
                for entry in input_data:
                    json_line = json.dumps(entry)
                    file.write(json_line + "\n")

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

    def parse_batch_output_line(self, line: str) -> List[Dict[str, Any]]:
        raw_output = json.loads(line)

        split_custom_id = raw_output["custom_id"].split("_")
        options = ast.literal_eval(split_custom_id[-1])
        question_id = int(split_custom_id[-2])
        subject = "_".join(split_custom_id[:-2])

        _, test_df = self._load_dataframes(subject)

        raw_answer = raw_output["response"]["body"]["choices"][0]["message"]["content"]
        match = re.search(self.ANSWER_PATTERN_MULTICHOICE, raw_answer)
        letter_answer = match.group(1) if match else None
        mapped_letter_answer = (
            options[string.ascii_uppercase.index(letter_answer)]
            if letter_answer
            else None
        )

        results = [
            {
                "question_id": question_id,
                "question_text": test_df.iloc[question_id, 0],
                "options_text": test_df.iloc[question_id, 1:5].to_list(),
                "options": options,
                "mapped_letter_answer": mapped_letter_answer,
                "rationale": raw_answer,
                "solution": test_df.iloc[question_id, -1],
            }
        ]

        return results

    def gen_results(self) -> None:
        accuracies = self._calculate_accuracies()
        subjects_statistics = self._calculate_subjects_statistics()
        self._plot_accuracy_distribution(accuracies)
        self._generate_question_distribution_plots(subjects_statistics)
        self._print_statistics_results(accuracies, subjects_statistics)

    def _calculate_accuracies(self) -> pd.DataFrame:
        accuracies = {}
        print("Calculating accuracies...")
        for file_name in tqdm(os.listdir(self.get_parsed_batch_outputs_dir())):
            subject, df = self._load_parsed_batch_output(file_name)
            accuracies[subject] = self._calculate_subject_accuracies(df)
        return pd.DataFrame(accuracies)

    def _load_parsed_batch_output(self, file_name: str) -> Tuple[str, pd.DataFrame]:
        file_path = os.path.join(self.get_parsed_batch_outputs_dir(), file_name)
        df = pd.read_csv(file_path)
        subject = "_".join(file_name.split("_")[:-4])
        return subject, df

    def _calculate_subject_accuracies(self, df: pd.DataFrame) -> Dict[str, float]:
        subject_accuracies = {}
        for option in df["options"].unique():
            accuracy = (
                df[df["options"] == option]
                .apply(lambda x: x["mapped_letter_answer"] == x["solution"], axis=1)
                .mean()
            )
            subject_accuracies[option] = accuracy
        return subject_accuracies

    def _plot_accuracy_distribution(self, accuracies: pd.DataFrame) -> None:
        sorted_accuracies = accuracies.loc[:, accuracies.loc["('A', 'B', 'C', 'D')"].sort_values().index]  # type: ignore
        fig = self._create_violin_plot(sorted_accuracies)
        self._add_no_shuffling_accuracy_trace(fig, accuracies)
        fig.show()
        fig.write_html(
            os.path.join(self.get_results_dir(), "accuracy_distribution.html")
        )

    def _create_violin_plot(self, sorted_accuracies: pd.DataFrame) -> go.Figure:
        return px.violin(
            sorted_accuracies,
            x=sorted_accuracies.columns,
            points="all",
            hover_name=sorted_accuracies.index,
            title="Accuracy Distribution per Subject",
            labels={"value": "Accuracy", "variable": "Subject"},
        )

    def _add_no_shuffling_accuracy_trace(
        self, fig: go.Figure, accuracies: pd.DataFrame
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=accuracies.loc["('A', 'B', 'C', 'D')"],
                y=accuracies.columns,
                mode="markers",
                marker=dict(color="red", size=10, symbol="circle"),
                name="No-Shuffling Accuracy",
            )
        )

    def _calculate_subjects_statistics(self) -> Dict[str, pd.DataFrame]:
        results = {}
        print("Calculating statistics...")
        for file_name in tqdm(os.listdir(self.get_parsed_batch_outputs_dir())):
            subject, df = self._load_parsed_batch_output(file_name)
            grouped_df = (
                df.groupby(["question_id", "mapped_letter_answer"])
                .size()
                .reset_index(name="count")
            )  # type: ignore

            grouped_df["entropy"] = grouped_df.groupby("question_id")[
                "count"
            ].transform(lambda x: entropy(x / x.sum(), base=2))
            majority_answer_df = grouped_df.loc[
                grouped_df.groupby("question_id")["count"].idxmax()
            ]
            grouped_df = grouped_df.merge(
                majority_answer_df[["question_id", "mapped_letter_answer"]],
                on="question_id",
                suffixes=("", "_majority"),
            )
            grouped_df.rename(
                columns={"mapped_letter_answer_majority": "majority_answer"},
                inplace=True,
            )
            grouped_df_sorted = grouped_df.sort_values(
                by=["entropy", "majority_answer", "question_id"]
            ).reset_index(drop=True)
            grouped_df_sorted["question_id"] = grouped_df_sorted["question_id"].astype(
                str
            )
            results[subject] = grouped_df_sorted
        return results

    def _generate_question_distribution_plots(
        self, subjects_statistics: Dict[str, pd.DataFrame]
    ) -> None:
        for subject, grouped_df in subjects_statistics.items():
            self._plot_question_distribution(grouped_df, f"{subject}.csv")

    def _plot_question_distribution(
        self, grouped_df: pd.DataFrame, file_name: str
    ) -> None:
        fig = px.bar(
            grouped_df,
            x="question_id",
            y="count",
            color="mapped_letter_answer",
            title="Distribution of Answers per Question",
            barmode="stack",
            category_orders={
                "mapped_letter_answer": ["A", "B", "C", "D"],
                "question_id": grouped_df["question_id"].tolist(),
            },
            labels={"question_id": "Question ID", "count": "Count"},
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.T10,
        )
        fig.write_html(
            os.path.join(
                self.get_results_dir(), f"{file_name.removesuffix('.csv')}.html"
            )
        )

    def _print_statistics_results(
        self, accuracies: pd.DataFrame, subjects_statistics: Dict[str, pd.DataFrame]
    ) -> None:
        results = {}
        for subject, grouped_df in subjects_statistics.items():
            results[subject] = {
                "original_accuracy": accuracies.loc["('A', 'B', 'C', 'D')", subject],
                "mean_entropy": grouped_df["entropy"].mean(),
            }
        results = pd.DataFrame(results).T
        results = results.sort_values(by="original_accuracy")
        print(results)
        print(f"Correlation between accuracy and entropy: {results.corr().iloc[0, 1]}")
