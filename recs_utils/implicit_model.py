import warnings
from typing import Optional

import numpy as np
import polars as pl
from implicit.recommender_base import RecommenderBase as ImplicitBase
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseItemSim
from .matrix_ops import interactions_to_csr_matrix
from .neg_samples import select_pos_samples


def to_light_fm_feature(row_info: dict, entity_key: str):
    return (row_info[entity_key], row_info["features"])


class ImplicitRecommender(BaseItemSim):
    def __init__(self,
                 inner_model: ImplicitBase,
                 rating_col_name: Optional[str] = None,
                 user_col: str = "user_id",
                 item_column: str = "item_id",
                 dt_column: str = "start_date"):
        super().__init__(user_col=user_col, item_column=item_column, dt_column=dt_column)
        self._inner_model = inner_model
        self._train_matrix = None
        self._user_encoder = LabelEncoder()
        self._item_encoder = LabelEncoder()
        self._rating_col_name = rating_col_name
        self._schema = None

    @property
    def model_name(self):
        return self._inner_model.__class__.__name__

    def fit(self,
            user_item_interactions: pl.DataFrame,
            user_features=None,
            item_features=None,
            progress: bool = False,
            fitted_user_encoder: Optional[LabelEncoder] = None,
            fitted_item_encoder: Optional[LabelEncoder] = None,
            **kwargs):
        user_item_interactions = select_pos_samples(user_item_interactions)

        if fitted_user_encoder is not None:
            self._user_encoder = fitted_user_encoder

        if fitted_item_encoder is not None:
            self._item_encoder = fitted_item_encoder

        self._user_encoder.fit(user_item_interactions.get_column(self.user_column).unique())
        self._item_encoder.fit(user_item_interactions.get_column(self.item_column).unique())

        self._train_matrix = self.get_train_matrix(user_item_interactions)
        self._schema = user_item_interactions.schema
        self._inner_model.fit(self._train_matrix, show_progress=progress)

    def get_train_matrix(self, user_item_interactions: pl.DataFrame):
        return interactions_to_csr_matrix(
            user_item_interactions,
            self._user_encoder,
            self._item_encoder,
            weight_col=self._rating_col_name)

    def most_similar_items(self, item_ids: pl.Series, n_neighbours: int):
        assert self._schema is not None, "Train first fit(...)"
        assert self._item_encoder is not None

        item_ids_numpy = item_ids.to_numpy()
        col_ids = self._item_encoder.transform(item_ids_numpy)

        most_similar_col_ids, scores = self._inner_model.similar_items(col_ids, N=n_neighbours)
        most_sim_item_ids = self._item_encoder.inverse_transform(most_similar_col_ids.reshape(-1))

        sim_items = pl.DataFrame(
            {
                self.item_column: item_ids_numpy.repeat(n_neighbours),
                "similar_item_id": most_sim_item_ids,
                "score": scores.reshape(-1)
            },
            schema={
                self.item_column: self._schema["item_id"],
                "similar_item_id": self._schema["item_id"],
                "score": pl.Float32
            }
        )

        return sim_items

    def recommend(self,
                  user_item_interactions: pl.DataFrame,
                  num_recs_per_user: int = 10,
                  user_features: Optional[pl.DataFrame] = None,
                  item_features: Optional[pl.DataFrame] = None,
                  filter_already_liked_items: bool = True,
                  **kwargs):
        assert self._train_matrix is not None, "Train first"
        assert self._schema is not None, "Train first"
        assert self._user_encoder is not None
        assert self._item_encoder is not None

        user_ids_numpy = user_item_interactions.get_column(self.user_column).unique().to_numpy()

        local_user_ids = self._user_encoder.transform(user_ids_numpy)

        col_ids, _ = self._inner_model.recommend(
            local_user_ids,
            self._train_matrix[local_user_ids, :],
            N=num_recs_per_user,
            filter_already_liked_items=filter_already_liked_items)

        col_ids = col_ids.reshape(-1)
        mask = col_ids == -1
        num_missings = np.count_nonzero(mask)

        if num_missings > 0:
            warnings.warn(
                f"Detected {num_missings} missing values for recommendations. Fill it with random items", UserWarning)

        col_ids[mask] = np.random.randint(0, self._train_matrix.shape[1], size=num_missings)
        item_ids = self._item_encoder.inverse_transform(col_ids.reshape(-1))

        recs = pl.DataFrame(
            {
                self.user_column: user_ids_numpy.repeat(num_recs_per_user),
                self.item_column: item_ids
            },
            schema={k: v for k, v in self._schema.items()
                    if k in (self.user_column, self.item_column)}
        )

        recs = recs.with_columns(
            (pl.col(self.user_column).cumcount().over(self.user_column) + 1).alias("rank"))
        return recs


class ImplicitRecommenderWithInterConf(ImplicitRecommender):
    def __init__(self,
                 inner_model: ImplicitBase,
                 rating_col_name: str,
                 user_col: str = "user_id",
                 item_column: str = "item_id",
                 dt_column: str = "start_date",
                 rating_quant_level: int = 5):
        super().__init__(inner_model, rating_col_name, user_col, item_column, dt_column)
        self._progress_quant_level = rating_quant_level

    def get_train_matrix(self, user_item_interactions: pl.DataFrame):
        user_item_interactions_with_quant_progress = user_item_interactions.with_columns(
            (pl.col(self._rating_col_name) // self._progress_quant_level) * self._progress_quant_level)

        return interactions_to_csr_matrix(
            user_item_interactions_with_quant_progress,
            self._user_encoder,
            self._item_encoder,
            weight_col=self._rating_col_name)
