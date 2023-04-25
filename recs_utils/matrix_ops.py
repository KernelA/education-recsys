from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse


def interactions_to_csr_matrix(interactions: pd.DataFrame,
                               users_mapping: Dict[int, int],
                               items_mapping: Dict[int, int],
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None):
    if weight_col is None:
        weights = np.ones(len(interactions), dtype=np.float32)
    else:
        weights = interactions[weight_col].to_numpy(dtype=np.float32)

    interaction_matrix = sparse.csr_matrix(
        (
            weights,
            (
                interactions.index.get_level_values(user_col).map(users_mapping.get).to_numpy(),
                interactions.index.get_level_values(item_col).map(items_mapping.get).to_numpy()
            )
        )
    )
    return interaction_matrix
