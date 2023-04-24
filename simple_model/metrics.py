import numpy as np
import pandas as pd


def join_true_pred_and_preprocess(true_pred: pd.DataFrame, recommendations: pd.DataFrame):
    data = true_pred.join(recommendations, how="left")
    data["item_count_per_user"] = data.groupby(level="user_id")["rank"].transform(np.size)
    data.sort_values(by=["user_id", "rank"], inplace=True)
    return data


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
