from typing import Optional, Sequence, NamedTuple

import polars as pl
import numpy as np
from catboost import FeaturesData


class ListwiseTrainRankingInfo(NamedTuple):
    target: np.ndarray
    group_ids: np.ndarray
    timestamps: np.ndarray
    features: FeaturesData


class PairWiseRankingInfo(NamedTuple):
    target: np.ndarray
    pairs: np.ndarray
    group_ids: np.ndarray
    timestamps: np.ndarray
    features: FeaturesData


class ListWiseTestRankingInfo(NamedTuple):
    features: FeaturesData


def add_features(interaction: pl.DataFrame, features: pl.DataFrame, join_cols: Sequence[str]):
    return interaction.lazy()\
        .join(features.lazy(), on=join_cols, how="inner").collect()


def add_group_ids(interactions_with_features: pl.DataFrame, group_by: Sequence[str], group_id_col_name: str = "group_id"):
    return interactions_with_features.lazy().join(
        interactions_with_features.lazy().groupby(group_by).agg(
            [
                pl.lit(0).alias(group_id_col_name)
            ]
        ).with_columns(pl.col(group_id_col_name).cumcount()),
        how="inner",
        on=group_by).sort(group_id_col_name).collect()


def convert_to_features_data(interactions_with_features: pl.DataFrame, exclude_columns: Optional[Sequence[str]] = None) -> FeaturesData:
    if exclude_columns is None:
        exclude_columns = []

    cat_features_names = [col for col, col_dtype in interactions_with_features.schema.items() if
                          (col_dtype.is_(pl.Categorical) or col_dtype.is_(pl.Utf8)) and col not in exclude_columns]

    num_features_names = [col for col, col_dtype in interactions_with_features.schema.items() if col_dtype
                          in pl.NUMERIC_DTYPES and col not in cat_features_names and col not in exclude_columns]

    num_features = None

    if num_features_names:
        num_features = interactions_with_features.select(
            pl.col(num_features_names)).to_numpy().astype(np.float32)

    cat_features = None

    if cat_features_names:
        cat_features = interactions_with_features.select(pl.col(cat_features_names)).to_numpy()

    return FeaturesData(
        num_feature_data=num_features,
        cat_feature_data=cat_features,
        num_feature_names=num_features_names if num_features_names else None,
        cat_feature_names=cat_features_names if cat_features_names else None)


def features_target_pairwise_ranking(interactions_with_features: pl.DataFrame, target_col: str = "target", time_col: str = "start_date", user_id_col: str = "user_id", pair_id_col: str = "pair_id", item_id_col: str = "item_id", group_id_col: str = "group_id") -> PairWiseRankingInfo:
    interactions_with_features = interactions_with_features.sort(group_id_col)
    pairs_indices = interactions_with_features.lazy().with_row_count().select(
        pl.col("row_nr"),
        pl.col(target_col),
        pl.col(pair_id_col)
    ).sort(pair_id_col, target_col, descending=[False, True]).select(pl.col("row_nr")).collect().to_numpy().reshape(-1, 2)

    target = interactions_with_features.select(
        pl.col(target_col)).to_series().to_numpy().astype(np.float32)
    timestamps = interactions_with_features.select(pl.col(time_col)).to_series().to_numpy()
    group_ids = interactions_with_features.select(pl.col(group_id_col)).to_series().to_numpy()

    interactions_with_features = interactions_with_features.drop(
        [target_col, group_id_col, pair_id_col, user_id_col, item_id_col, time_col])

    return PairWiseRankingInfo(target, pairs_indices, group_ids, timestamps, convert_to_features_data(interactions_with_features))


def features_target_ranking(interactions_with_features: pl.DataFrame, target_col: str = "target", group_id_col: str = "group_id", time_col: str = "start_date", user_id_col: str = "user_id", item_id_col: str = "item_id") -> ListwiseTrainRankingInfo:
    interactions_with_features = interactions_with_features.sort(group_id_col)
    target = interactions_with_features.select(
        pl.col(target_col)).to_series().to_numpy().astype(np.float32)
    group_id = interactions_with_features.select(
        pl.col(group_id_col)).to_series().to_numpy().astype(np.int32)
    timestamps = interactions_with_features.select(pl.col(time_col)).to_series().to_numpy()

    interactions_with_features = interactions_with_features.drop(
        [target_col, group_id_col, user_id_col, item_id_col, time_col])

    return ListwiseTrainRankingInfo(target, group_id, timestamps, convert_to_features_data(interactions_with_features))
