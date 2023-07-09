from typing import Dict, Union, Type, Optional

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


def _to_sparse_matrix(matrix_class: Union[Type[sparse.csc_matrix], Type[sparse.csr_matrix]],
                      interactions: pl.DataFrame,
                      users_encoder: LabelEncoder,
                      items_encoder: LabelEncoder,
                      user_col: str = 'user_id',
                      item_col: str = 'item_id',
                      weight_col: Optional[str] = None):

    if weight_col is None:
        weights = np.ones(len(interactions), dtype=np.float32)
    else:
        weights = interactions.get_column(weight_col).to_numpy().astype(np.float32)

    interaction_matrix = matrix_class(
        (
            weights,
            (
                users_encoder.transform(interactions.get_column(user_col).to_numpy()),
                items_encoder.transform(interactions.get_column(item_col).to_numpy())
            )
        )
    )

    return interaction_matrix


def interactions_to_csr_matrix(interactions: pl.DataFrame,
                               users_encoder: LabelEncoder,
                               items_encoder: LabelEncoder,
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None):
    return _to_sparse_matrix(sparse.csr_matrix, interactions, users_encoder, items_encoder, user_col=user_col, item_col=item_col, weight_col=weight_col)


def interactions_to_csc_matrix(interactions: pl.DataFrame,
                               users_encoder: LabelEncoder,
                               items_encoder: LabelEncoder,
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None):
    return _to_sparse_matrix(sparse.csc_matrix, interactions, users_encoder, items_encoder, user_col=user_col, item_col=item_col, weight_col=weight_col)
