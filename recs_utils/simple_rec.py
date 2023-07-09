from abc import ABC, abstractmethod
import datetime
from typing import Optional

import polars as pl
import numpy as np

from .base_model import BaseRecommender


class PopularRecommender(BaseRecommender):
    def __init__(self,
                 max_K: int = 100,
                 days: int = 30,
                 user_col: str = "user_id",
                 item_column: str = "item_id",
                 dt_column: str = "start_date"):
        super().__init__(user_col, item_column, dt_column)
        self.max_K = max_K
        self.days = days
        self.recommendations: pl.DataFrame = pl.DataFrame()

    @property
    def model_name(self):
        return "most_popular"

    def fit(self, interactions: pl.DataFrame, user_features=None, item_features=None):
        min_date = interactions.select(pl.col(self.dt_column).max())[
            0, 0] - datetime.timedelta(days=self.days)

        self.recommendations = interactions.lazy()\
            .filter(pl.col(self.dt_column) > min_date)\
            .groupby(self.item_column)\
            .count()\
            .top_k(self.max_K, by="count")\
            .select(pl.col(self.item_column))\
            .with_columns(
            (pl.col(self.item_column).cumcount() + 1).alias("rank"))\
            .collect()\


    def recommend(self, user_item_interactions: pl.DataFrame, num_recs_per_user: int = 10, user_features=None, item_features=None):
        assert self.recommendations is not None, "fit(...) first"

        recs_items = self.recommendations.head(n=num_recs_per_user)
        uniq_users = user_item_interactions.get_column(self.user_column).unique().to_numpy()

        recs = pl.DataFrame(
            {
                self.user_column: uniq_users.repeat(len(recs_items)),
                self.item_column: np.tile(recs_items.get_column(self.item_column).to_numpy(), len(uniq_users)),
                "rank": np.tile(recs_items.get_column("rank").to_numpy(), len(uniq_users)),
            }
        )

        return recs


class PopularRecommenderPerAge(BaseRecommender):
    def __init__(self,
                 max_K: int = 100,
                 days: int = 30,
                 user_col: str = "user_id",
                 item_column: str = 'item_id',
                 dt_column: str = 'start_date'):
        super().__init__(user_col, item_column, dt_column)
        self.max_K = max_K
        self.days = days
        self.recommendations_per_age: pl.DataFrame = pl.DataFrame()
        self._top_n = PopularRecommender(max_K, days, user_col, item_column, dt_column)

    @property
    def model_name(self):
        return "most_popular_per_age"

    def fit(self, user_item_interactions: pl.DataFrame, user_features: pl.DataFrame, item_features=None):
        min_date = user_item_interactions.select(pl.col(self.dt_column).max())[
            0, 0] - datetime.timedelta(days=self.days)

        self.recommendations_per_age = user_item_interactions.lazy()\
            .filter(pl.col(self.dt_column) > min_date)\
            .join(user_features.lazy().select(pl.col(self.user_column), pl.col("age")), on="user_id", how="inner")\
            .groupby(["age", self.item_column])\
            .count().collect()\
            .groupby("age")\
            .apply(lambda group: group.top_k(k=self.max_K, by="count"))\
            .with_columns((pl.col("age").cumcount().over("age") + 1).alias("rank"))

        self._top_n.fit(user_item_interactions)

    def recommend(self, user_item_interactions: pl.DataFrame, num_recs_per_user: int = 10, user_features: Optional[pl.DataFrame] = None, item_features=None):
        assert self.recommendations_per_age is not None
        assert user_features is not None

        user_ids = user_item_interactions.select(pl.col(self.user_column).unique())

        users_with_known_ages = user_ids.join(user_features.select(
            pl.col(self.user_column), pl.col("age")), on=self.user_column, how="inner")

        users_with_unknown_ages = user_ids.join(user_features.select(
            pl.col(self.user_column)), on=self.user_column, how="anti")

        user_features_query = users_with_known_ages.lazy()
        user_features_query = user_features_query.select(
            pl.col(self.user_column),
            pl.col("age")
        )\
            .join(self.recommendations_per_age.lazy(), on="age", how="left")

        recs_with_known_age = user_features_query\
            .select(
                pl.col(self.user_column),
                pl.col(self.item_column),
                pl.col("rank")
            )\
            .filter(pl.col("rank") <= num_recs_per_user).collect()

        recs_with_unknown_age = user_features_query.filter(
            pl.col("rank").is_null()).select(pl.col(self.user_column).unique()).collect().vstack(users_with_unknown_ages)

        if not recs_with_unknown_age.is_empty():
            recs_with_unknown_ages = self._top_n.recommend(
                recs_with_unknown_age, num_recs_per_user=num_recs_per_user)

        all_recs = recs_with_known_age.vstack(recs_with_unknown_ages)

        need_to_fill = all_recs.lazy().groupby(self.user_column).count().with_columns(
            (num_recs_per_user - pl.col("count")).alias("new_items")
        ).filter(pl.col("new_items") > 0).collect()

        if not need_to_fill.is_empty():
            num_to_predict = need_to_fill.get_column("new_items").max()
            add_recommends = self._top_n.recommend(
                need_to_fill.select(pl.col(self.user_column).unique()),
                num_recs_per_user=num_to_predict
            )

            add_recommends = add_recommends.join(
                need_to_fill,
                on=self.user_column,
                how="inner"
            ).with_columns(
                (pl.col("rank") + pl.col("count")).alias("rank")
            )

            all_recs = all_recs.vstack(
                add_recommends.select(
                    pl.col(self.user_column),
                    pl.col(self.item_column),
                    pl.col("rank")
                )
            ).filter(pl.col("rank") <= num_recs_per_user)

        return all_recs
