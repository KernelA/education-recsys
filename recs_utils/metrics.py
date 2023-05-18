from collections import defaultdict
from typing import Callable

import polars as pl
import pandas as pd
from tqdm.auto import tqdm

from .implicit_model import ModelRecommender

USER_ID_COL = "user_id"
ITEM_ID_COL = "item_id"


def model_cross_validate(interactions: pl.DataFrame, item_features: pl.DataFrame, user_features: pl.DataFrame, folds, model_factory: Callable[[], ModelRecommender], n: int):
    validation_results = []

    for num_fold, (train_idx, test_idx, info) in enumerate(tqdm(folds), 1):
        train = interactions.join(train_idx, on=[USER_ID_COL, ITEM_ID_COL], how="inner")
        test = interactions.join(test_idx, on=[USER_ID_COL, ITEM_ID_COL], how="inner")
        train_item_features = item_features.filter(pl.col(ITEM_ID_COL).is_in(
            train.select(pl.col(ITEM_ID_COL).unique()).to_series()))
        train_user_features = user_features.filter(pl.col(USER_ID_COL).is_in(
            train.select(pl.col(USER_ID_COL).unique()).to_series()))

        model: ModelRecommender = model_factory()
        model.fit(train, progress=False, train_item_features=train_item_features,
                  train_user_features=train_user_features)
        pred_recs: pl.DataFrame = model.recommend(test.select(pl.col(USER_ID_COL).unique()).to_series(), n)

        metrics = compute_metrics(test, pred_recs, n)
        metrics = metrics.with_columns(
            pl.lit(model.model_name()).alias("model"),
            pl.lit(info["Start date"]).alias("start_date"),
            pl.lit(info["End date"]).alias("end_date"),
            pl.lit(num_fold).alias("fold")
        )
        validation_results.append(metrics)

    cv_results: pl.DataFrame = pl.concat(validation_results)
    cv_results = cv_results.with_columns(pl.col("model").cast(pl.Categorical).alias("model"))
    return cv_results.sort("start_date"), model


def join_true_pred_and_preprocess(true_pred: pl.DataFrame, recommendations: pl.DataFrame):
    data = true_pred.lazy().join(recommendations.lazy(), how="left", on=[USER_ID_COL, ITEM_ID_COL])
    data = data.join(
        data.groupby(USER_ID_COL).agg(pl.count().alias("item_count_per_user")),
        on=USER_ID_COL
    )

    data = data.with_columns(pl.col("rank").fill_null(float("nan")))
    return data.collect()


def compute_metrics(true_pred: pl.DataFrame, recs: pl.DataFrame, max_k: int):
    joined_data = join_true_pred_and_preprocess(true_pred, recs)

    metrics = defaultdict(list)

    for k in range(1, max_k + 1, 1):
        metrics["name"].append(f"prec@{k}")
        value, joined_data = precision_at_k(joined_data, k)
        metrics["value"].append(value)

        metrics["name"].append(f"recall@{k}")
        value, joined_data = precision_at_k(joined_data, k)
        metrics["value"].append(value)

    metrics["name"].append("MRR")
    metrics["value"].append(mean_reciprocal_rank(joined_data))
    metrics["name"].append("MAP")
    metrics["value"].append(mean_average_prec(joined_data))

    return pl.from_dict(metrics)


def compute_hit_at_k(joined_data: pl.DataFrame, k: int):
    metric_name = f"hit@{k}"

    if metric_name not in joined_data.columns:
        joined_data = joined_data.with_columns(
            joined_data.select((pl.col("rank") <= k).fill_null(False).alias(metric_name))
        )

    return joined_data


def precision_at_k(joined_data: pl.DataFrame, k: int):
    metric_name = f"hit@{k}"

    joined_data = compute_hit_at_k(joined_data, k)

    joined_data = joined_data.with_columns(
        joined_data.select(
            (pl.col(metric_name) / k).alias(f"{metric_name}/{k}")
        )
    )

    return (joined_data.select(pl.col(f"{metric_name}/{k}").sum()) / joined_data.select(pl.col(USER_ID_COL).n_unique()))[0, 0], joined_data


def recall_at_k(joined_data: pl.DataFrame, k: int):
    joined_data = compute_hit_at_k(joined_data, k)
    return (joined_data.select(pl.col(f"hit@{k}") / pl.col("item_count_per_user")).sum() / joined_data.select(pl.col("user_id").n_unique()))[0, 0], joined_data


def mean_reciprocal_rank(joined_data: pl.DataFrame):
    return joined_data.lazy().select(
        [
            pl.col(USER_ID_COL),
            (1 / pl.col("rank")).alias("inv_rank").fill_nan(0.0)
        ]).groupby(USER_ID_COL).agg(
        [
            pl.max("inv_rank")
        ]
    ).select("inv_rank").mean().collect()[0, 0]


def mean_average_prec(joined_data: pl.DataFrame):
    cumulative_rank = joined_data.lazy()\
        .select(
            ((pl.col(USER_ID_COL).cumcount().over(USER_ID_COL) + 1) / pl.col("rank"))
        .fill_nan(0.0)
        .alias("cum_rank"),
            pl.col("item_count_per_user")
    )

    users_count = joined_data.select(pl.col(USER_ID_COL).n_unique())[0, 0]

    return cumulative_rank.select(pl.col("cum_rank") / pl.col("item_count_per_user")).sum().collect()[0, 0] / users_count
