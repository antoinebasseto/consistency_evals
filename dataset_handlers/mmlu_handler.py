import ast
import json
import os
import re
import string
from typing import Any, Dict, List, Tuple

import numpy as np
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
        if bool(re.match(r"^\d+$", split_custom_id[-1])):
            split_custom_id = split_custom_id[:-1]
        options = ast.literal_eval(split_custom_id[-1])
        question_id = int(split_custom_id[-2])
        subject = "_".join(split_custom_id[:-2])

        _, test_df = self._load_dataframes(subject)

        raw_answer = raw_output["response"]["body"]["choices"][0]["message"]["content"]
        match = re.search(self.ANSWER_PATTERN_MULTICHOICE, raw_answer)
        letter_answer = match.group(1) if match else None

        mapped_letter_answer = None
        if letter_answer is not None:
            mapped_letter_index = string.ascii_uppercase.index(letter_answer)  # type: ignore
            if mapped_letter_index < len(options):
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
        if self.experiment == "withdrawing":
            self._gen_results_withdrawing()

        if self.experiment == "shuffling" or self.experiment == "control":
            accuracies = self._calculate_accuracies()
            subjects_statistics = self._calculate_subjects_statistics()

            for file_name in tqdm(
                os.listdir(
                    os.path.join(
                        self.get_parsed_batch_outputs_dir(), "..", "withdrawing"
                    )
                )
            ):
                _, df = self._load_parsed_batch_output(
                    os.path.join("..", "withdrawing", file_name)
                )
                subject = "_".join(file_name.split("_")[:-4])
                df = df[["question_id", "options", "mapped_letter_answer"]]
                max_value_counts = df.groupby("question_id")[
                    "mapped_letter_answer"
                ].agg(lambda x: x.value_counts().max())

                _, test_df = self._load_dataframes(subject)
                test_df["withdrawing_max_value_counts"] = max_value_counts

                print(test_df)

                test_df.iloc[
                    subjects_statistics[subject]["question_id"].drop_duplicates()
                ].to_csv(
                    os.path.join(
                        self.get_results_dir(), "ordered_by_entropy", f"{subject}.csv"
                    ),
                )
                # df_grouped = df.groupby(["question_id", "mapped_letter_answer"]).max()

                # print(df_grouped)
                # _, test_df = self._load_dataframes(df_subject)
                # test_df.iloc[subjects_statistics[df_subject]["question_id"].drop_duplicates()]

            self._plot_accuracy_distribution(accuracies)
            self._generate_question_distribution_plots(subjects_statistics)
            self._print_statistics_results(accuracies, subjects_statistics)

    def _gen_results_withdrawing(self) -> None:
        results = []
        for file_name in tqdm(os.listdir(self.get_parsed_batch_outputs_dir())):
            subject, df = self._load_parsed_batch_output(file_name)
            df_filled = df[["question_id", "options", "mapped_letter_answer"]].fillna(
                "E"
            )
            df_filled["mapped_letter_answer"] = df_filled["mapped_letter_answer"].map(
                {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
            )
            df_pivot = df_filled.pivot(
                columns="question_id", index="options", values="mapped_letter_answer"
            )
            sorted_columns = sorted(
                df_pivot.columns,
                key=lambda col: (
                    df_pivot[col].value_counts().max(),
                    df_pivot[col].nunique(),
                    df_pivot[col].value_counts().idxmax(),
                    df_pivot[col].value_counts().idxmin(),
                ),
                reverse=True,
            )
            df_sorted = df_pivot[sorted_columns]
            fig = px.imshow(
                df_sorted.to_numpy(),
                x=list(map(str, sorted_columns)),
                y=df_sorted.index,
                color_continuous_scale=list(
                    zip(
                        np.linspace(0, 1, num=5),
                        px.colors.qualitative.T10[:4] + ["white"],
                    )
                ),
            )
            fig.update_layout(
                coloraxis_showscale=False,
                width=1920,
                height=1080,
                xaxis=dict(
                    showticklabels=False,
                ),
                xaxis_title="Question ID",
                font=dict(size=10),
            )
            fig.write_html(os.path.join(self.get_results_dir(), f"{subject}.html"))

            results.append(
                {
                    "subject": subject,
                    "percentage_question_with_striclty_more_than_two_unique_answers": (
                        (df_sorted.nunique(axis=0) > 2).sum() / df_sorted.shape[1]
                    ),
                    "percentage_question_where_no_answer_is_chosen_three_times": (
                        (
                            df_sorted.apply(lambda x: x.value_counts().max(), axis=0)
                            < 3
                        ).sum()
                        / df_sorted.shape[1]
                    ),
                }
            )
        results = pd.DataFrame(results).sort_values(
            by="percentage_question_where_no_answer_is_chosen_three_times"
        )
        print(results)
        results.to_csv(
            os.path.join(self.get_results_dir(), f"results_{self.experiment}.csv"),
            index=False,
        )

        results_shuffling = pd.read_csv(
            os.path.join(
                self.get_results_dir(), "..", "shuffling", "results_shuffling.csv"
            )
        )

        merged_results = results.merge(
            results_shuffling,
            on="subject",
            suffixes=("_withdrawing", "_shuffling"),
        )

        print(merged_results)

        print(
            f"Mean weighted entropy: {sum(merged_results['mean_entropy'] * merged_results['n_questions']) / sum(merged_results['n_questions'])}"
        )
        print(
            f"Mean weighted accuracy: {sum(merged_results['original_accuracy'] * merged_results['n_questions']) / sum(merged_results['n_questions'])}"
        )
        print(
            f"Mean weighted percentage_question_with_striclty_more_than_two_unique_answers: {sum(merged_results['percentage_question_with_striclty_more_than_two_unique_answers'] * merged_results['n_questions']) / sum(merged_results['n_questions'])}"
        )
        print(
            f"Mean weighted percentage_question_where_no_answer_is_chosen_three_times: {sum(merged_results['percentage_question_where_no_answer_is_chosen_three_times'] * merged_results['n_questions']) / sum(merged_results['n_questions'])}"
        )
        print(merged_results.drop("subject", axis=1).corr())

        merged_results.to_csv(
            os.path.join(self.get_results_dir(), "..", f"results_merged.csv"),
            index=False,
        )
        merged_results.drop("subject", axis=1).corr().to_csv(
            os.path.join(self.get_results_dir(), "..", f"correlation.csv"),
            index=False,
        )

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
        if self.experiment == "shuffling":
            self._add_no_shuffling_accuracy_trace(fig, accuracies)
        fig.write_html(
            os.path.join(self.get_results_dir(), "accuracy_distribution.html")
        )

    def _create_violin_plot(self, sorted_accuracies: pd.DataFrame) -> go.Figure:
        return px.violin(
            sorted_accuracies,
            x=sorted_accuracies.columns,
            # points="all",
            hover_name=sorted_accuracies.index,
            # title="Accuracy Distribution per Subject",
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
        fig.update_layout(showlegend=False)
        fig.update_layout(height=1400, yaxis=dict(tickfont=dict(size=12)))

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
            # title="Distribution of Answers per Question",
            barmode="stack",
            category_orders={
                "mapped_letter_answer": ["A", "B", "C", "D"],
                "question_id": grouped_df["question_id"].tolist(),
            },
            labels={"question_id": "Question ID", "count": "Count"},
            # text_auto=True,
            color_discrete_sequence=px.colors.qualitative.T10,
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis=dict(showticklabels=False))
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
        for file_name in tqdm(os.listdir(self.get_parsed_batch_outputs_dir())):
            subject, df = self._load_parsed_batch_output(file_name)
            results[subject]["n_questions"] = df["question_id"].nunique()
        results = pd.DataFrame(results).T
        results = results.sort_values(by="original_accuracy")
        print(results)
        results.to_csv(
            os.path.join(self.get_results_dir(), f"results_{self.experiment}.csv"),
            index_label="subject",
        )
        print(
            f"Mean weighted entropy: {sum(results['mean_entropy'] * results['n_questions']) / sum(results['n_questions'])}"
        )
        print(
            f"Mean weighted accuracy: {sum(results['original_accuracy'] * results['n_questions']) / sum(results['n_questions'])}"
        )
        print(f"Correlation between accuracy and entropy: {results.corr().iloc[0, 1]}")
