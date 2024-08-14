from dataset_handlers.custom_case_vignettes_handler import CustomCaseVignettesHandler
from dataset_handlers.mmlu_handler import MMLUHandler
from dataset_handlers.nejm_handler import NEJMHandler


def get_dataset_handler(dataset_name: str, engine: str, experiment: str, ai_type: str):
    if dataset_name == "mmlu":
        return MMLUHandler(engine, experiment, ai_type)
    if dataset_name == "nejm":
        return NEJMHandler(engine, experiment, ai_type)
    if dataset_name == "custom":
        return CustomCaseVignettesHandler(engine, experiment, ai_type)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
