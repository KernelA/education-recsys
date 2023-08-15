import pathlib

import fasttext
import hydra
import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download

from recs_utils.constants import ITEM_ID_COL
from recs_utils.log_set import init_logging


@hydra.main(config_path="configs", config_name="process_features", version_base="1.3")
def main(process_features):
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())

    data_dir = orig_cwd / process_features.orig_features_dir
    users = pl.read_parquet(data_dir / "users.parquet")
    users = users.select(pl.col("user_id", "age", "sex")).to_dummies(["age", "sex"])

    users.write_parquet("user_features.parquet")

    del users
    items = pl.read_parquet(data_dir / "items.parquet")
    items = items.with_columns(
        pl.col("title").str.strip().str.to_lowercase().alias("title"),
        pl.col("genres").cast(str).fill_null("unknown").str.strip().str.to_lowercase().str.split(
            ",").arr.sort().arr.join(" ").fill_null("unknown").alias("genres")
    )

    model_path = hf_hub_download(repo_id="kernela/fasttext-ru-vectors-dim-100",
                                 filename="ru-vectors-dim-100.bin", cache_dir=orig_cwd / "hub-cache")
    model = fasttext.load_model(model_path)

    genres_embeddings = []

    for row in items.iter_rows(named=True):
        genres_embeddings.append(model.get_sentence_vector(row["genres"]).reshape(1, -1))

    item_features = pl.DataFrame(np.concatenate(genres_embeddings, axis=0)
                                 ).hstack([items.get_column(ITEM_ID_COL)])
    item_features.write_parquet("items.parquet")


if __name__ == "__main__":
    init_logging(log_config="log_settings.yaml")
    main()
