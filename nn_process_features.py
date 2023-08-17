import pathlib

import hydra
import polars as pl

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

    item_features = items.lazy().select(
        pl.col("genres").str.split(","),
        pl.col("item_id")).explode("genres").with_columns(pl.lit(1, dtype=pl.Int8).alias("genre_feature")).collect().pivot(
        values="genre_feature",
        index="item_id",
        columns="genres",
        aggregate_function="max"
    ).fill_null(0)

    item_features.write_parquet("items.parquet")


if __name__ == "__main__":
    init_logging(log_config="log_settings.yaml")
    main()
