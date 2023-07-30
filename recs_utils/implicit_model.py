import warnings
from typing import Iterable, Optional

import numpy as np
import polars as pl
from implicit.recommender_base import RecommenderBase as ImplicitBase
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseItemSim
from .matrix_ops import interactions_to_csr_matrix


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
            progress: bool = False,
            user_features=None,
            item_features=None,
            fitted_user_encoder: Optional[LabelEncoder] = None,
            fitted_item_encoder: Optional[LabelEncoder] = None):
        if fitted_user_encoder is not None:
            self._user_encoder = fitted_user_encoder

        if fitted_item_encoder is not None:
            self._item_encoder = fitted_item_encoder

        self._user_encoder.fit(user_item_interactions.get_column(self.user_column).unique())
        self._item_encoder.fit(user_item_interactions.get_column(self.item_column).unique())

        self._train_matrix = interactions_to_csr_matrix(
            user_item_interactions, self._user_encoder, self._item_encoder, weight_col=self._rating_col_name)
        self._schema = user_item_interactions.schema
        self._inner_model.fit(self._train_matrix, show_progress=progress)

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
                  filter_already_liked_items: bool = True):
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


# class LightFMRecommender(BaseItemSim):
#     def __init__(self, model: LightFM, num_epoch: int, num_threads: int):
#         super().__init__({}, {}, {})
#         self._model = model
#         self.num_epoch = num_epoch
#         self._train_user_features = None
#         self._train_item_features = None
#         self._known_items_per_user_id = {}
#         self.num_threads = num_threads
#         self._schema = None

#     def model_name(self):
#         return self._model.__class__.__name__

#     def fit(self, interactions: pl.DataFrame,
#             progress: bool = True,
#             train_user_features: Optional[pl.DataFrame] = None,
#             train_item_features: Optional[pl.DataFrame] = None):
#         assert train_user_features is not None
#         assert train_item_features is not None

#         self._schema = interactions.schema

#         self._known_items_per_user_id = {
#             info["user_id"]: info["item_id"] for info in interactions.lazy().select(pl.col("user_id"), pl.col("item_id")).groupby('user_id').agg(
#                 [
#                     pl.list("item_id")
#                 ]
#             ).collect().to_dicts()}

#         dataset = Dataset()

#         dataset.fit(interactions.select(pl.col("user_id")).unique().to_series(),
#                     interactions.select(pl.col("item_id")).unique().to_series())

#         train_user_features = train_user_features.clone()

#         train_user_features = train_user_features.with_columns(
#             pl.col("age").cast(str).fill_null("age_unknown").cast(pl.Categorical).alias("age"),
#             pl.col("sex").cast(str).fill_null("sex_unknown").cast(pl.Categorical).alias("sex")
#         )

#         age_features = train_user_features.select(pl.col("age").unique()).to_series().to_list()
#         sex_features = train_user_features.select(pl.col("sex").unique()).to_series().to_list()

#         users_features = age_features + sex_features
#         dataset.fit_partial(user_features=users_features)

#         train_item_features = train_item_features.clone()
#         train_item_features = train_item_features.with_columns(
#             pl.col("genres").cast(str).fill_null(
#                 "genre_unknown").cast(pl.Categorical).alias("genres")
#         )

#         genres = train_item_features.select(pl.col("genres").cast(
#             str).str.split(',').explode().unique()).to_series().to_list()
#         dataset.fit_partial(item_features=genres)

#         lightfm_mapping = dataset.mapping()

#         self.user_mapping = lightfm_mapping[0]
#         self.item_mapping = lightfm_mapping[2]
#         self.inv_item_mapping = {v: k for k, v in self.item_mapping.items()}

#         train_mat, _ = dataset.build_interactions(interactions.select(
#             pl.col("user_id"),
#             pl.col("item_id")
#         ).to_numpy()
#         )

#         train_user_features = train_user_features.with_columns(
#             pl.concat_list(pl.col("age").cast(str),
#                            pl.col("sex").cast(str)).alias("features")
#         )

#         known_users_mask = train_user_features.select(pl.col("user_id").is_in(
#             interactions.select(pl.col("user_id").unique()).to_series())).to_series()

#         train_user_features = dataset.build_user_features(
#             map(lambda x: to_light_fm_feature(x, "user_id"), train_user_features.filter(
#                 known_users_mask).select(pl.col("user_id"), pl.col('features')).to_dicts())
#         )

#         self._train_user_features = train_user_features

#         train_item_features = train_item_features.with_columns(
#             pl.col("genres").cast(str).str.split(",").alias("features")
#         )

#         known_items_mask = train_item_features.select(pl.col("item_id").is_in(
#             interactions.select("item_id").unique().to_series())).to_series()

#         train_item_features = dataset.build_item_features(
#             map(lambda x: to_light_fm_feature(x, "item_id"), train_item_features.filter(known_items_mask).select(
#                 pl.col("item_id"),
#                 pl.col("features")
#             ).to_dicts()))

#         self._train_item_features = train_item_features

#         self._model.fit(train_mat, user_features=self._train_user_features, item_features=self._train_item_features, epochs=self.num_epoch,
#                         num_threads=self.num_threads)

#     def recommend(self, user_ids: Iterable[int], n: int):
#         assert self._schema is not None, "Train first"

#         user_ids = tuple(user_ids)
#         local_user_ids = np.array(tuple(map(self.user_mapping.get, user_ids)), dtype=np.int32)

#         num_uniq_users = len(local_user_ids)
#         local_item_ids = list(self.item_mapping.values())
#         num_uniq_items = len(local_item_ids)

#         local_item_ids *= num_uniq_users
#         local_user_ids = local_user_ids.repeat(num_uniq_items)

#         predicted_scores = self._model.predict(
#             local_user_ids,
#             local_item_ids,
#             item_features=self._train_item_features,
#             user_features=self._train_user_features,
#             num_threads=self.num_threads)

#         predicted_scores = predicted_scores.reshape((num_uniq_users, num_uniq_items))

#         final_item_ids = []

#         for row_num, user_id in enumerate(user_ids):
#             additional_N = len(self._known_items_per_user_id.get(user_id, []))
#             total_N = n + additional_N
#             top_item_local_ids = np.argpartition(
#                 predicted_scores[row_num], -np.arange(total_N))[-total_N:][::-1]

#             final_recs = list(map(self.inv_item_mapping.get, top_item_local_ids))

#             if additional_N > 0:
#                 filter_items = self._known_items_per_user_id[user_id]
#                 final_recs = [item for item in final_recs if item not in filter_items][:n]

#             final_item_ids.extend(final_recs)

#         recs = pl.DataFrame(
#             {
#                 "user_id": np.asarray(user_ids).repeat(n),
#                 "item_id": final_item_ids
#             },
#             schema={k: v for (k, v) in self._schema.items() if k in ("user_id", "item_id")}
#         )

#         return recs.with_columns((pl.col("user_id").cumcount().over("user_id") + 1).alias("rank"))
