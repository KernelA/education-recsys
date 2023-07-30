from collections import defaultdict
from typing import Callable

import pandas as pd
import polars as pl
from scipy.spatial import distance
from tqdm.auto import tqdm

from .base_model import BaseRecommender

USER_ID_COL = "user_id"
ITEM_ID_COL = "item_id"


def _compute_diversity_value(condensed_distance_matrix, num_items: int):
    return 2 / (num_items * (num_items - 1)) * condensed_distance_matrix.sum()


def intra_list_diversity_hamming_per_user(recommendations: pl.DataFrame, item_features: pl.DataFrame):
    user_ids = []
    diversity_values = []

    for user_id, group in recommendations.groupby(USER_ID_COL):
        user_ids.append(user_id)
        num_items = group.get_column("rank").max()
        diversity_values.append(
            _compute_diversity_value(
                distance.pdist(
                    group.select(ITEM_ID_COL).join(item_features, on=ITEM_ID_COL).select(
                        pl.all().exclude(ITEM_ID_COL)).to_numpy(), metric="hamming"), num_items
            )
        )

    return pl.DataFrame({USER_ID_COL: user_ids, "intra_list_div": diversity_values},
                        schema={USER_ID_COL: recommendations.schema[USER_ID_COL], "intra_list_div": pl.Float32})


def mean_inverse_user_freq_per_user(recommendations: pl.DataFrame, train_interactions: pl.DataFrame):
    num_users = train_interactions.get_column(USER_ID_COL).n_unique()
    num_users_per_item = train_interactions.groupby(
        ITEM_ID_COL).agg(pl.n_unique(USER_ID_COL).alias("num_inter_users"))

    num_users_per_item = num_users_per_item.vstack(
        recommendations.lazy().select(pl.col(ITEM_ID_COL).unique()).join(
            num_users_per_item.lazy(), how="anti", on=ITEM_ID_COL).with_columns(
            pl.lit(1).cast(num_users_per_item.dtypes[num_users_per_item.columns.index("num_inter_users")]).alias("num_inter_users")).collect()
    )

    miuf_per_user = recommendations.lazy().join(num_users_per_item.lazy(), on=ITEM_ID_COL).select(
        pl.col(USER_ID_COL), pl.col("rank"), (pl.col(
            "num_inter_users") / num_users).log(base=2).alias("inv_user_freq")
    ).groupby(USER_ID_COL).agg(
        [
            (-pl.mean("inv_user_freq")).alias("miuf")
        ]).collect()

    return miuf_per_user


def serendipity_per_user(recommendations: pl.DataFrame, train_interactions: pl.DataFrame, test_interactions: pl.DataFrame):
    item_pop_rank = train_interactions.lazy().select(pl.col(ITEM_ID_COL).value_counts(sort=True)).select(
        pl.col(ITEM_ID_COL).struct.field(ITEM_ID_COL),
        pl.col(ITEM_ID_COL).struct.field("counts").rank(
            method="ordinal", descending=True).alias("item_pop_rank")
    ).collect()

    joined_data = recommendations.lazy().join(test_interactions.lazy().with_columns(
        pl.lit(1).alias("rel")), how="left", on=[USER_ID_COL, ITEM_ID_COL]).with_columns(
        pl.col("rel").fill_null(0)
    )

    return joined_data.join(
        item_pop_rank.lazy(), on=ITEM_ID_COL).with_columns(
        (pl.max(
            pl.col("item_pop_rank") - pl.col("rank"), 0) * pl.col("rel")).alias("seren")
    ).groupby(USER_ID_COL).agg(pl.mean("seren")).collect()


def model_cross_validate(
        user_item_interactions: pl.DataFrame,
        item_features: pl.DataFrame,
        user_features: pl.DataFrame,
        folds,
        model_factory: Callable[[], BaseRecommender],
        num_recs: int):
    validation_results = []

    if not folds:
        raise RuntimeError("Empty folds detected")

    for num_fold, (train_idx, test_idx, info) in enumerate(tqdm(folds), 1):
        train_interactions = user_item_interactions.join(
            train_idx, on=[USER_ID_COL, ITEM_ID_COL], how="inner")
        test_interactions = user_item_interactions.join(
            test_idx, on=[USER_ID_COL, ITEM_ID_COL], how="inner")

        train_item_features = item_features.filter(pl.col(ITEM_ID_COL).is_in(
            train_interactions.get_column(ITEM_ID_COL).unique()))
        train_user_features = user_features.filter(pl.col(USER_ID_COL).is_in(
            train_interactions.get_column(USER_ID_COL).unique()))

        model: BaseRecommender = model_factory()
        model.fit(
            train_interactions,
            item_features=train_item_features,
            user_features=train_user_features)

        test_item_features = item_features.filter(pl.col(ITEM_ID_COL).is_in(
            test_interactions.get_column(ITEM_ID_COL).unique()))
        test_user_features = user_features.filter(pl.col(USER_ID_COL).is_in(
            test_interactions.get_column(USER_ID_COL).unique()))

        pred_recs: pl.DataFrame = model.recommend(
            test_interactions,
            num_recs,
            user_features=test_user_features,
            item_features=test_item_features)

        metrics = compute_metrics(test_interactions, pred_recs, num_recs)

        metrics = metrics.with_columns(
            pl.lit(model.model_name).alias("model"),
            pl.lit(info["Start date"]).alias("start_date"),
            pl.lit(info["End date"]).alias("end_date"),
            pl.lit(num_fold).alias("fold")
        )
        validation_results.append(metrics)

    cv_results: pl.DataFrame = pl.concat(validation_results)
    cv_results = cv_results.with_columns(pl.col("model").cast(pl.Categorical))
    return cv_results.sort("start_date"), model


def join_true_pred_and_preprocess(true_pred: pl.DataFrame, recommendations: pl.DataFrame):
    data = true_pred.lazy().join(recommendations.lazy(),
                                 how="left", on=[USER_ID_COL, ITEM_ID_COL])
    data = data.join(
        data.groupby(USER_ID_COL).agg(pl.count().alias("item_count_per_user")),
        on=USER_ID_COL
    )

    data = data.with_columns(pl.col("rank").fill_null(float("nan")))
    return data.collect()


def compute_metrics(true_pred: pl.DataFrame, recs: pl.DataFrame, max_k: int, with_separate_k_col: bool = False):
    joined_data = join_true_pred_and_preprocess(true_pred, recs)

    metrics = defaultdict(list)

    def add_metric_name(name, k, metrics):
        if with_separate_k_col:
            metrics["name"].append(name)
            metrics["k"].append(k)
        else:
            metrics["name"].append(f"{name}@{k}")

    for k in range(1, max_k + 1, 1):
        add_metric_name("prec", k, metrics)
        value, joined_data = precision_at_k(joined_data, k)
        metrics["value"].append(value)

        add_metric_name("recall", k, metrics)
        value, joined_data = precision_at_k(joined_data, k)
        metrics["value"].append(value)

    add_metric_name("MRR", max_k, metrics)
    metrics["value"].append(mean_reciprocal_rank(joined_data))

    add_metric_name("MAP", max_k, metrics)
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
    return (joined_data.select(pl.col(f"hit@{k}") / pl.col("item_count_per_user")).sum() / joined_data.select(pl.col(USER_ID_COL).n_unique()))[0, 0], joined_data


def mean_reciprocal_rank(joined_data: pl.DataFrame):
    mrr_eval = joined_data.lazy()

    return mrr_eval.select(
        [
            pl.col(USER_ID_COL),
            (1 / pl.col("rank")).alias("inv_rank").fill_nan(0.0)
        ]).groupby(USER_ID_COL).agg(
        [
            pl.max("inv_rank")
        ]
    ).select("inv_rank").mean().collect()[0, 0]


def mean_average_prec(joined_data: pl.DataFrame):
    """k taken into account in the joined_data
    """
    cumulative_rank = joined_data.lazy()

    cumulative_rank = cumulative_rank\
        .select(
            ((pl.col(USER_ID_COL).cumcount().over(USER_ID_COL) + 1) / pl.col("rank"))
            .fill_nan(0.0)
            .alias("cum_rank"),
            pl.col("item_count_per_user")
        )

    users_count = joined_data.get_column(USER_ID_COL).n_unique()

    return cumulative_rank.select(pl.col("cum_rank") / pl.col("item_count_per_user")).sum().collect()[0, 0] / users_count
