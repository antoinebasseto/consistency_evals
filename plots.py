import pandas as pd
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer


def plot_text_clusters(df: pd.DataFrame, text_col: str, save_path: str):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(df[text_col].tolist())

    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)

    df["x"] = reduced_embeddings[:, 0]  # type: ignore
    df["y"] = reduced_embeddings[:, 1]  # type: ignore

    # Convert answer_id to string for plotly to use categorical colors
    df["answer_id"] = df["answer_id"].astype(str)
    fig = px.scatter(df, x="x", y="y", color="answer_id", hover_data={"claim": True})
    fig.show()
    fig.write_image(save_path)


def plot_answer_distribution(
    df: pd.DataFrame, options: str, answer_col: str, save_path: str
):
    fig = px.histogram(df, x=answer_col, category_orders={answer_col: options})
    fig.show()
    fig.write_image(save_path)
