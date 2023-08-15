import polars as pl

from ..constants import ITEM_ID_COL, USER_ID_COL


def compute_stat(interactions: pl.DataFrame):
    avg_user_rating = interactions.lazy().filter(pl.col("rating").is_not_nan()).groupby(
        USER_ID_COL).agg(pl.mean("rating").alias("avg_user_rating"))

    agg_stat_by_user = interactions.lazy().groupby(USER_ID_COL).agg(
        [
            pl.col(ITEM_ID_COL).count().alias("num_interacted_items"),
            pl.mean("progress").alias("mean_user_progress"),
        ]
    ).join(avg_user_rating, on=USER_ID_COL, how="left").with_columns(pl.col("avg_user_rating").fill_null(0)).collect()

    return agg_stat_by_user


def get_pos_neg_samples(interactions: pl.DataFrame, stat: pl.DataFrame):
    cols = interactions.columns

    neg_samples = interactions.lazy().join(stat.lazy(), on=USER_ID_COL).filter(
        (pl.col("rating").is_not_nan() & (pl.col("rating") < pl.col("avg_user_rating"))) |
        (pl.col("rating").is_nan() & (pl.col("progress") < pl.col("mean_user_progress"))))\
        .with_columns(pl.lit(0, dtype=pl.Int8).alias("target")).select(pl.col(cols), pl.col("target")).collect()

    pos_samples = interactions.join(
        neg_samples, on=[USER_ID_COL, ITEM_ID_COL], how="anti").with_columns(pl.lit(1, dtype=neg_samples.schema["target"]).alias("target"))

    # If no positive samples for user then threat negative as positives
    pos_samples = pos_samples.vstack(neg_samples.join(
        pos_samples, on=[USER_ID_COL], how="anti").with_columns(pl.lit(1, dtype=pl.Int8).alias("target")))

    return pos_samples, neg_samples
