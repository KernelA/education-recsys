from typing import Dict

import numpy as np
import polars as pl
from scipy import sparse


def interactions_to_csr_matrix(interactions: pl.DataFrame,
                               users_mapping: Dict[int, int],
                               items_mapping: Dict[int, int],
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None):
    if weight_col is None:
        weights = np.ones(len(interactions), dtype=np.float32)
    else:
        weights = interactions.select(weight_col).to_series().to_numpy(dtype=np.float32)

    interaction_matrix = sparse.csr_matrix(
        (
            weights,
            (
                interactions.select(pl.col(user_col).apply(users_mapping.get)).to_series().to_numpy(),
                interactions.select(pl.col(item_col).apply(items_mapping.get)).to_series().to_numpy()
            )
        )
    )
    return interaction_matrix
