from typing import Dict, Tuple

import pandas as pd


def get_direct_and_inv_mapping(interactions: pd.DataFrame, index_name: str) -> Tuple[dict, dict]:
    users_inv_mapping = dict(enumerate(interactions.index.unique(index_name)))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    return users_mapping, users_inv_mapping
