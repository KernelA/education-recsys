from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .implicit_model import ImplicitRecommender
from .matrix_ops import interactions_to_csr_matrix


def implicit_cross_validate(interactions: pd.DataFrame, item_mapping, folds, model_factory: Callable[[], ImplicitRecommender], n: int):
    validation_results = []

    for num_fold, (train_idx, test_idx, info) in enumerate(tqdm(folds), 1):
        train: pd.DataFrame = interactions.loc[train_idx]
        test: pd.DataFrame = interactions.loc[test_idx]

        model: ImplicitRecommender = model_factory()
        train_matrix = interactions_to_csr_matrix(train, model.user_mapping, item_mapping)
        model.fit(train_matrix, progress=False)
        pred_recs = model.recommend(test.index.get_level_values("user_id").unique().to_numpy(), n)

        metrics = compute_metrics(test, pred_recs, n).transpose()
        metrics["model"] = model.model_name()
        metrics["start_date"] = info["Start date"]
        metrics["end_date"] = info["End date"]
        metrics["fold"] = num_fold
        validation_results.append(metrics)

    cv_results = pd.concat(validation_results)

    return cv_results.sort_values("start_date"), model


def join_true_pred_and_preprocess(true_pred: pd.DataFrame, recommendations: pd.DataFrame):
    data = true_pred.join(recommendations, how="left")
    data["item_count_per_user"] = data.groupby(level="user_id")["rank"].transform(np.size)
    data = data.sort_values(by=["user_id", "rank"])
    return data


def compute_metrics(true_pred: pd.DataFrame, recs: pd.DataFrame, max_k: int):
    joined_data = join_true_pred_and_preprocess(true_pred, recs)

    metrics = defaultdict(list)

    for k in range(1, max_k + 1, 1):
        metrics["name"].append(f"prec@{k}")
        metrics["value"].append(precision_at_k(joined_data, k))

        metrics["name"].append(f"recall@{k}")
        metrics["value"].append(recall_at_k(joined_data, k))

    metrics["name"].append("MRR")
    metrics["value"].append(mean_reciprocal_rank(joined_data))
    metrics["name"].append("MAP")
    metrics["value"].append(mean_average_prec(joined_data))

    return pd.DataFrame.from_dict(metrics).set_index("name")


def compute_hit_at_k(joined_data: pd.DataFrame, k: int):
    metric_name = f"hit@{k}"

    if metric_name not in joined_data.columns:
        joined_data[metric_name] = joined_data["rank"] <= k


def precision_at_k(joined_data: pd.DataFrame, k: int):
    metric_name = f"hit@{k}"
    compute_hit_at_k(joined_data, k)

    joined_data[f"{metric_name}/{k}"] = joined_data[metric_name] / k
    return joined_data[f"{metric_name}/{k}"].sum() / joined_data.index.get_level_values("user_id").nunique()


def recall_at_k(joined_data: pd.DataFrame, k: int):
    compute_hit_at_k(joined_data, k)
    return (joined_data[f"hit@{k}"] / joined_data["item_count_per_user"]).sum() / joined_data.index.get_level_values("user_id").nunique()


def mean_reciprocal_rank(joined_data: pd.DataFrame):
    inverse_rank_per_user = (1 / joined_data["rank"]).groupby("user_id").max()
    return inverse_rank_per_user.fillna(0).mean()


def mean_average_prec(joined_data: pd.DataFrame):
    cumulative_rank = joined_data.groupby(level='user_id').cumcount() + 1
    cumulative_rank = cumulative_rank / joined_data["rank"]
    users_count = joined_data.index.get_level_values("user_id").nunique()
    return (cumulative_rank / joined_data["item_count_per_user"]).sum() / users_count
