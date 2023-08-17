from typing import Optional

import numpy as np
import polars as pl
from lightfm import LightFM
from lightfm.data import Dataset
from tqdm.auto import tqdm

from .base_model import BaseRecommender


def to_light_fm_feature(row_info: dict, entity_key: str):
    return (row_info[entity_key], row_info["features"])


class LightFMRecommender(BaseRecommender):
    def __init__(self,
                 model: LightFM,
                 num_epoch: int,
                 num_threads: int,
                 user_col: str = "user_id",
                 item_column: str = "item_id",
                 dt_column: str = "start_date",
                 progress_column: Optional[str] = None,
                 progress_quant_level: int = 5,
                 ):
        super().__init__(user_col=user_col, item_column=item_column, dt_column=dt_column)
        self._model = model
        self.num_epoch = num_epoch
        self._train_user_features = None
        self._train_item_features = None
        self._user_item_uniq_train_inter = pl.DataFrame()
        self.num_threads = num_threads
        self._progress_column = progress_column
        self._schema = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.inv_item_mapping = {}
        self._progress_quant_level = progress_quant_level

    @property
    def model_name(self):
        return self._model.__class__.__name__

    def fit(self,
            user_item_interactions: pl.DataFrame,
            user_features: Optional[pl.DataFrame] = None,
            item_features: Optional[pl.DataFrame] = None,
            progress: bool = False,
            **kwargs):
        assert user_features is not None
        assert item_features is not None

        self._schema = user_item_interactions.schema

        self._user_item_uniq_train_inter = user_item_interactions.select(
            pl.col(self.user_column), pl.col(self.item_column)).unique()

        dataset = Dataset()

        dataset.fit(user_item_interactions.get_column(self.user_column).unique(),
                    user_item_interactions.get_column(self.item_column).unique())

        user_features = user_features.with_columns(
            pl.col("age").cast(str).fill_null("age_unknown"),
            pl.col("sex").cast(str).fill_null("sex_unknown")
        )

        age_feature_names = user_features.get_column("age").unique().to_list()
        sex_feature_names = user_features.get_column("sex").unique().to_list()

        dataset.fit_partial(user_features=age_feature_names + sex_feature_names)

        del age_feature_names
        del sex_feature_names

        item_features = item_features.with_columns(
            pl.col("genres").cast(str).fill_null("genre_unknown")
        )

        genres = item_features.get_column("genres").str.split(',').explode().unique().to_list()
        dataset.fit_partial(item_features=genres)

        self.user_mapping, _, self.item_mapping, _ = dataset.mapping()
        self.inv_item_mapping = {v: k for k, v in self.item_mapping.items()}

        inter_cols = [pl.col(self.user_column), pl.col(self.item_column)]

        if self._progress_column is not None:
            user_item_interactions = user_item_interactions.with_columns(
                (pl.col(self._progress_column) // self._progress_quant_level) * self._progress_quant_level)

            inter_cols.append(pl.col(self._progress_column))

        _, train_weight_matrix = dataset.build_interactions(user_item_interactions.select(
            inter_cols
        ).iter_rows(named=False)
        )

        user_features = user_features.with_columns(
            pl.concat_list(pl.col("age"),
                           pl.col("sex")).alias("features")
        )

        known_users_mask = user_features.get_column(self.user_column).is_in(
            user_item_interactions.get_column(self.user_column).unique())

        user_features = dataset.build_user_features(
            map(lambda x: to_light_fm_feature(x, self.user_column), user_features.filter(
                known_users_mask).select(pl.col(self.user_column), pl.col('features')).iter_rows(named=True))
        )

        del known_users_mask

        self._train_user_features = user_features

        item_features = item_features.with_columns(
            pl.col("genres").str.split(",").alias("features")
        )

        known_items_mask = item_features.get_column(self.item_column).is_in(
            user_item_interactions.get_column(self.item_column).unique())

        item_features = dataset.build_item_features(
            map(lambda x: to_light_fm_feature(x, self.item_column), item_features.filter(known_items_mask).select(
                pl.col(self.item_column),
                pl.col("features")
            ).iter_rows(named=True)))

        del known_items_mask

        self._train_item_features = item_features

        self._model.fit(train_weight_matrix,
                        user_features=self._train_user_features,
                        item_features=self._train_item_features,
                        epochs=self.num_epoch,
                        num_threads=self.num_threads,
                        verbose=progress)

    def recommend(self,
                  user_item_interactions: pl.DataFrame,
                  num_recs_per_user: int = 10,
                  user_features: Optional[pl.DataFrame] = None,
                  item_features: Optional[pl.DataFrame] = None,
                  user_batch_size: Optional[int] = None,
                  is_prediction_progress: bool = False,
                  filter_already_liked: bool = True,
                  ** kwargs):
        assert self._schema is not None, "Train first fit(...)"

        user_ids = user_item_interactions.get_column(self.user_column).unique()
        local_user_ids = user_ids.apply(self.user_mapping.get).to_numpy()

        num_uniq_users = len(local_user_ids)
        local_item_ids = np.array(tuple(self.item_mapping.values()))
        num_uniq_items = len(local_item_ids)

        if user_batch_size is None:
            user_batch_size = num_uniq_users

        max_interacted_items = self._user_item_uniq_train_inter.groupby(
            self.user_column).count().get_column("count").max()

        total_items_per_user = num_recs_per_user + max_interacted_items

        start_index = 0

        predicted_local_item_ids = []

        pred_progress = tqdm(total=len(local_user_ids),
                             disable=not is_prediction_progress, mininterval=3)

        while start_index < len(local_user_ids):
            local_user_ids_batch = local_user_ids[start_index: start_index + user_batch_size]
            start_index += user_batch_size
            num_users_in_batch = len(local_user_ids_batch)
            local_user_ids_batch = local_user_ids_batch.repeat(num_uniq_items)
            local_item_ids_batch = np.tile(local_item_ids, num_users_in_batch)

            predicted_scores = self._model.predict(
                local_user_ids_batch,
                local_item_ids_batch,
                item_features=self._train_item_features,
                user_features=self._train_user_features,
                num_threads=self.num_threads)

            del local_user_ids_batch
            del local_item_ids_batch

            predicted_scores = predicted_scores.reshape((num_users_in_batch, num_uniq_items))
            local_batch_item_ids = np.flip(np.argpartition(predicted_scores, -np.arange(total_items_per_user), axis=1)
                                           [:, -total_items_per_user:], axis=1)

            predicted_local_item_ids.append(local_batch_item_ids.reshape(-1))
            del local_batch_item_ids
            pred_progress.update(n=num_users_in_batch)

        predicted_local_item_ids = np.concatenate(predicted_local_item_ids)

        recs = pl.DataFrame(
            {
                self.user_column: np.asarray(user_ids).repeat(total_items_per_user),
                self.item_column: predicted_local_item_ids
            },
            schema={k: v for (k, v) in self._schema.items()
                    if k in (self.user_column, self.item_column)}
        )

        recs = recs.with_columns(pl.col(self.item_column).apply(
            self.inv_item_mapping.get).cast(self._schema[self.item_column]))

        if filter_already_liked:
            recs = recs.join(self._user_item_uniq_train_inter, on=[
                self.user_column, self.item_column], how="anti")

        recs = recs.with_columns(
            (pl.col(self.user_column).cumcount().over(self.user_column) + 1).alias("rank"))

        return recs.filter(pl.col("rank") <= num_recs_per_user)
