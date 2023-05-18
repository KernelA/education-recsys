from typing import Dict, Tuple

import polars as pl


def get_direct_and_inv_mapping(interactions: pl.DataFrame, index_name: str) -> Tuple[dict, dict]:
    users_inv_mapping = dict(enumerate(interactions.select(
        pl.col(index_name).unique()).to_series()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    return users_mapping, users_inv_mapping
