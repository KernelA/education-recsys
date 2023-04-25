from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    def __init__(self, max_K: int = 100, days: int = 30, user_col: str = "user_id", item_column: str = 'item_id', dt_column: str = 'start_date'):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.user_column = user_col

    @abstractmethod
    def fit(self, interactions: pd.DataFrame, user_features=None, item_features=None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, users=None, N: int = 10):
        raise NotImplementedError()


class PopularRecommender(BaseRecommender):
    def __init__(self, max_K: int = 100, days: int = 30,  user_col: str = "user_id", item_column: str = 'item_id', dt_column: str = 'start_date'):
        super().__init__(max_K, days, user_col, item_column, dt_column)
        self.recommendations = []

    def fit(self, interactions: pd.DataFrame, user_features=None, item_features=None):
        min_date = interactions[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = interactions.loc[interactions[self.dt_column] > min_date].reset_index(
        )[self.item_column].value_counts().head(self.max_K).index.values.tolist()

    def predict(self, user_features: pd.DataFrame, N: int = 10):
        recs_items = self.recommendations[:N]

        recs = pd.DataFrame(
            {
                self.user_column: user_features.index.unique(self.user_column).to_numpy().repeat(len(recs_items)),
                self.item_column: recs_items * user_features.index.unique(self.user_column).size,
            }
        )

        recs["rank"] = recs.groupby(self.user_column).cumcount() + 1

        return recs.set_index([self.user_column, self.item_column])


class PopularRecommenderPerAge(BaseRecommender):
    def __init__(self, max_K: int = 100, days: int = 30,  user_col: str = "user_id", item_column: str = 'item_id', dt_column: str = 'start_date'):
        super().__init__(max_K, days, user_col, item_column, dt_column)
        self.recommendations = {}
        self._top_n = PopularRecommender(max_K, days, user_col, item_column, dt_column)

    def fit(self, interactions: pd.DataFrame, user_features, item_features=None):
        min_date = interactions[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)

        self.recommendations = pd.merge(
            interactions[interactions[self.dt_column] > min_date].reset_index(level=self.item_column)[
                self.item_column],
            user_features["age"], left_index=True, right_index=True
        )[["age", self.item_column]].value_counts().groupby("age", group_keys=False).nlargest(self.max_K)
        self._top_n.fit(interactions)

    def predict(self, user_features: pd.DataFrame, N: int = 10):
        missing_ages_mask = ~user_features["age"].isin(
            self.recommendations.index.get_level_values("age"))

        uniq_users = user_features.index.unique(self.user_column)

        recs = pd.DataFrame(
            data={self.item_column: [-1] * uniq_users.nunique()},
            index=uniq_users
        )

        if missing_ages_mask.any():
            rec_items = self._top_n.predict(user_features, N=N)\
                .reset_index(self.item_column).groupby(self.user_column).apply(lambda x: x[self.item_column].to_list())
            recs[self.item_column] = rec_items

            del rec_items

        ages_mask = ~missing_ages_mask

        rec_items = self.recommendations.loc[(user_features[ages_mask]["age"], slice(None))].nlargest(
            N).index.get_level_values(self.item_column).tolist()

        user_ids_with_features = user_features[ages_mask].index.get_level_values(self.user_column)
        recs.loc[user_ids_with_features, self.item_column] = [
            rec_items] * len(user_ids_with_features)

        recs = recs.explode(self.item_column)
        recs["rank"] = recs.groupby(self.user_column).cumcount() + 1

        return recs.reset_index().set_index([self.user_column, self.item_column])
