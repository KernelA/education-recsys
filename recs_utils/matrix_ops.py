from typing import Dict, Union, Type, Optional

import numpy as np
import polars as pl
from scipy import sparse


def _to_sparse_matrix(matrix_class: Union[Type[sparse.csc_matrix], Type[sparse.csr_matrix]],
                      interactions: pl.DataFrame,
                      users_mapping: Dict[int, int],
                      items_mapping: Dict[int, int],
                      user_col: str = 'user_id',
                      item_col: str = 'item_id',
                      weight_col: Optional[str] = None):

    if weight_col is None:
        weights = np.ones(len(interactions), dtype=np.float32)
    else:
        weights = interactions.select(weight_col).to_series().to_numpy().astype(np.float32)

    interaction_matrix = matrix_class(
        (
            weights,
            (
                interactions.select(pl.col(user_col).apply(
                    users_mapping.get)).to_series().to_numpy(),
                interactions.select(pl.col(item_col).apply(
                    items_mapping.get)).to_series().to_numpy()
            )
        )
    )

    return interaction_matrix


def interactions_to_csr_matrix(interactions: pl.DataFrame,
                               users_mapping: Dict[int, int],
                               items_mapping: Dict[int, int],
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None):
    return _to_sparse_matrix(sparse.csr_matrix, interactions, users_mapping, items_mapping, user_col=user_col, item_col=item_col, weight_col=weight_col)


def interactions_to_csc_matrix(interactions: pl.DataFrame,
                               users_mapping: Dict[int, int],
                               items_mapping: Dict[int, int],
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None):
    return _to_sparse_matrix(sparse.csc_matrix, interactions, users_mapping, items_mapping, user_col=user_col, item_col=item_col, weight_col=weight_col)
