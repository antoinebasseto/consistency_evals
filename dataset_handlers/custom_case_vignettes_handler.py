from dataset_handlers.dataset_handler import DatasetHandler
from dataset_handlers.nejm_handler import NEJMHandler


class CustomCaseVignettesHandler(NEJMHandler):
    def __init__(self, engine: str, experiment: str, ai_type: str) -> None:
        DatasetHandler.__init__(
            self, "custom_case_vignettes", engine, experiment, ai_type
        )
