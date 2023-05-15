from abc import ABC, abstractmethod
import datetime

import polars as pl
import pandas as pd
import numpy as np


class BaseRecommender(ABC):
    def __init__(self, max_K: int = 100, days: int = 30, user_col: str = "user_id", item_column: str = 'item_id', dt_column: str = 'start_date'):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.user_column = user_col

    @abstractmethod
    def fit(self, interactions: pl.DataFrame, user_features=None, item_features=None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, users=None, N: int = 10):
        raise NotImplementedError()


class PopularRecommender(BaseRecommender):
    def __init__(self, max_K: int = 100, days: int = 30,  user_col: str = "user_id", item_column: str = 'item_id', dt_column: str = 'start_date'):
        super().__init__(max_K, days, user_col, item_column, dt_column)
        self.recommendations: pl.DataFrame = None

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


    def predict(self, user_features: pl.DataFrame, N: int = 10):
        assert self.recommendations is not None

        recs_items = self.recommendations.head(n=N)
        uniq_users = user_features.select(pl.col(self.user_column).unique()).to_series().to_numpy()

        recs = pl.DataFrame(
            {
                self.user_column: uniq_users.repeat(len(recs_items)),
                self.item_column: np.tile(recs_items.select(pl.col(self.item_column)).to_series().to_numpy(), len(uniq_users)),
                "rank": np.tile(recs_items.select(pl.col("rank")).to_series().to_numpy(), len(uniq_users)),
            }
        )

        return recs


class PopularRecommenderPerAge(BaseRecommender):
    def __init__(self, max_K: int = 100, days: int = 30,  user_col: str = "user_id", item_column: str = 'item_id', dt_column: str = 'start_date'):
        super().__init__(max_K, days, user_col, item_column, dt_column)
        self.recommendations_per_age: pl.DataFrame = None
        self._top_n = PopularRecommender(max_K, days, user_col, item_column, dt_column)

    def fit(self, interactions: pl.DataFrame, user_features: pl.DataFrame, item_features=None):
        min_date = interactions.select(pl.col(self.dt_column).max())[
            0, 0] - datetime.timedelta(days=self.days)

        self.recommendations_per_age = interactions.lazy()\
            .filter(pl.col(self.dt_column) > min_date)\
            .join(user_features.lazy().filter(pl.col("age").is_not_null()), on="user_id", how="inner")\
            .groupby(["age", self.item_column])\
            .count()\
            .sort(["age", "count"], descending=True)\
            .groupby("age").head(n=self.max_K)\
            .with_columns((pl.col("age").cumcount().over("age") + 1).alias("rank"))\
            .collect()

        self._top_n.fit(interactions)

    def predict(self, user_features: pl.DataFrame, N: int = 10):
        assert self.recommendations_per_age is not None

        user_features_query = user_features.lazy()

        user_features_query = user_features_query.select(
            pl.col(self.user_column),
            pl.col("age")
        )\
            .join(self.recommendations_per_age.lazy(), on="age", how="left")

        recs_with_known_age = user_features_query.filter(pl.col("age").is_not_null())\
            .sort([self.user_column, "age", "rank"])\
            .groupby(self.user_column)\
            .head(N)\
            .select(
                pl.col(self.user_column),
                pl.col(self.item_column),
                pl.col("rank")
        )\
            .collect()

        users_with_unknown_ages = user_features_query.filter(pl.col("age").is_null())\
            .select(pl.col(self.user_column).unique())\
            .collect()

        recs_with_unknown_ages = self._top_n.predict(users_with_unknown_ages, N=N)

        need_to_fill = recs_with_known_age.groupby(self.user_column).count()\
            .with_columns(
                (N - pl.col("count")).alias("new_items")
        )\
            .filter(pl.col("new_items") > 0)

        if not need_to_fill.is_empty():
            num_to_predict = need_to_fill.select(pl.col("new_items").max())[0, 0]

            add_recommends = self._top_n.predict(
                need_to_fill.select(pl.col(self.user_column).unique()),
                N=num_to_predict
            )

            add_recommends = add_recommends.join(
                need_to_fill,
                on=self.user_column,
                how="inner"
            ).with_columns(
                (pl.col("rank") + pl.col("count")).alias("rank")
            )

            recs_with_known_age = recs_with_known_age.vstack(
                add_recommends.select(
                    pl.col(self.user_column),
                    pl.col(self.item_column),
                    pl.col("rank")
                )
            ).filter(pl.col("rank") <= N)

        return recs_with_known_age.vstack(recs_with_unknown_ages)
