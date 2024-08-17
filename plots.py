import os
from typing import List

import pandas as pd
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import LocalOutlierFactor


def plot_text_clusters(
    df: pd.DataFrame, text_col: str, color_col: str, save_path: str, file_name: str
):
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    embeddings = model.encode(df[text_col].tolist())

    clf = LocalOutlierFactor(n_neighbors=50)
    df["outlier"] = clf.fit_predict(embeddings)
    # df["outlier"] = clf.negative_outlier_factor_

    reducer = umap.UMAP(n_neighbors=50)
    reduced_embeddings = reducer.fit_transform(embeddings)

    df["x"] = reduced_embeddings[:, 0]  # type: ignore
    df["y"] = reduced_embeddings[:, 1]  # type: ignore

    fig = px.scatter(df, x="x", y="y", color="outlier", hover_data={"claim": True})
    # fig.show()
    fig.write_html(os.path.join(save_path, f"{file_name}.html"))


def plot_answer_distribution(
    df: pd.DataFrame,
    options: List[str],
    answer_col: str,
    save_path: str,
    file_name: str,
):
    fig = px.histogram(df, x=answer_col, category_orders={answer_col: options})
    # fig.show()
    fig.write_html(os.path.join(save_path, f"{file_name}.html"))
