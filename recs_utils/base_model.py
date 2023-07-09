from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

import polars as pl


class BaseRecommender(ABC):
    def __init__(self,
                 user_col: str = "user_id",
                 item_column: str = "item_id",
                 dt_column: str = "start_date"):
        self.item_column = item_column
        self.dt_column = dt_column
        self.user_column = user_col

    @abstractproperty
    def model_name(self):
        pass


    @abstractmethod
    def fit(self,
            user_item_interactions: pl.DataFrame,
            user_features: Optional[pl.DataFrame] = None,
            item_features: Optional[pl.DataFrame] = None,
            **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def recommend(self,
                  user_item_interactions: pl.DataFrame,
                  num_recs_per_user: int = 10,
                  user_features: Optional[pl.DataFrame] = None,
                  item_features: Optional[pl.DataFrame] = None,
                  **kwargs):
        raise NotImplementedError()


class BaseWithItemSim(BaseRecommender):
    @abstractmethod
    def most_similar_items(self, item_ids: pl.Series, n_neighbours: int):
        raise NotImplementedError()
