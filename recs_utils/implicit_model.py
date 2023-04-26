from typing import Iterable
from abc import ABC, abstractmethod

from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
import pandas as pd
from implicit.recommender_base import RecommenderBase
from scipy.sparse import csr_matrix

from .matrix_ops import interactions_to_csr_matrix


class ModelRecommender:
    def __init__(self, user_mapping, item_mapping, inv_item_mapping):
        self.user_mapping = user_mapping
        self.inv_item_mapping = inv_item_mapping
        self.item_mapping = item_mapping

    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def fit(self, interactions: pd.DataFrame, progress: bool = True, train_user_features = None, train_item_features = None):
        pass

    @abstractmethod
    def recommend(self, user_ids: Iterable[int], n: int):
        pass


class ImplicitRecommender(ModelRecommender):
    def __init__(self, model: RecommenderBase, user_mapping, item_mapping, inv_item_mapping):
        super().__init__(user_mapping, item_mapping, inv_item_mapping)
        self.model = model
        self._train_matrix = None

    def model_name(self):
        return self.model.__class__.__name__

    def fit(self, interactions: pd.DataFrame, progress: bool = True, train_user_features = None, train_item_features = None):
        self._train_matrix = interactions_to_csr_matrix(interactions, self.user_mapping, self.item_mapping)
        self.model.fit(self._train_matrix, show_progress=progress)

    def similiar_items(self, item_ids: Iterable[int], n: int):
        col_ids = list(map(self.item_mapping.get, item_ids))

        most_similiar_col_ids, scores = self.model.similar_items(col_ids, N=n)

        most_sim_item_ids = tuple(map(self.inv_item_mapping.get, most_similiar_col_ids.reshape(-1)))

        sim_items = pd.DataFrame(
            {
                "item_id": np.asarray(tuple(item_ids)).repeat(n),
                "similiar_item_id": most_sim_item_ids,
                "score": scores.reshape(-1)
            }
        )

        return sim_items

    def recommend(self, user_ids: Iterable[int], n: int):
        assert self._train_matrix is not None

        local_user_ids = list(map(self.user_mapping.get, user_ids))

        col_ids, _ = self.model.recommend(
            local_user_ids, self._train_matrix[local_user_ids, :], N=n, filter_already_liked_items=True)

        item_ids = tuple(map(self.inv_item_mapping.get, col_ids.reshape(-1)))

        recs = pd.DataFrame(
            {
                "user_id": np.asarray(tuple(user_ids)).repeat(n),
                "item_id": item_ids
            }
        )

        recs["rank"] = recs.groupby("user_id").cumcount() + 1

        return recs.set_index(["user_id", "item_id"])


class LightFMRecommender(ModelRecommender):
    def __init__(self, model: LightFM, num_epoch: int, num_threads: int):
        super().__init__({}, {}, {})
        self._model = model
        self.num_epoch = num_epoch
        self._train_user_features = None
        self._train_item_features = None
        self._known_items_per_user_id = {}
        self.num_threads = num_threads

    def model_name(self):
        return self._model.__class__.__name__


    def fit(self, interactions: pd.DataFrame, progress: bool = True, train_user_features=None, train_item_features=None):
        assert train_user_features is not None
        assert train_item_features is not None

        self._known_items_per_user_id = interactions.reset_index().groupby('user_id')["item_id"].apply(list).to_dict()

        dataset = Dataset()
        dataset.fit(interactions.index.get_level_values("user_id").unique(), interactions.index.get_level_values("item_id").unique())

        train_user_features = train_user_features.copy()
        train_user_features['age'] = train_user_features['age'].cat.add_categories('age_unknown')
        train_user_features['age'] = train_user_features['age'].fillna('age_unknown')
        age_features = train_user_features['age'].unique()


        assert -1 not in train_user_features["sex"]

        train_user_features['sex'] = train_user_features['sex'].fillna(-1).astype(int).astype(str).astype("category").cat.rename_categories({"-1": "sex_unknown"})
        sex_features = train_user_features['sex'].unique()

        users_features = np.append(age_features, sex_features)

        dataset.fit_partial(user_features=users_features)

        train_item_features = train_item_features.copy()
        train_item_features['genres'] = train_item_features['genres'].cat.add_categories('genre_unknown')
        train_item_features['genres'] = train_item_features['genres'].fillna('genre_unknown')
        genres = train_item_features['genres'].str.split(',').explode().unique().tolist()

        dataset.fit_partial(item_features=genres)

        lightfm_mapping = dataset.mapping()

        self.user_mapping = lightfm_mapping[0]
        self.item_mapping = lightfm_mapping[2]
        self.inv_item_mapping =  {v: k for k, v in self.item_mapping.items()}

        train_mat, _ = dataset.build_interactions(interactions.index)

        train_user_features['features'] = train_user_features[['age', 'sex']].astype(str).apply(lambda x: list(x), axis=1)

        known_users_mask = train_user_features.index.get_level_values("user_id").isin(interactions.index.get_level_values("user_id").unique())

        train_user_features = dataset.build_user_features(
            train_user_features.loc[known_users_mask]['features'].items()
        )

        self._train_user_features = train_user_features

        train_item_features['features'] = train_item_features['genres'].str.split(',')
        known_items_mask = train_item_features.index.get_level_values('item_id').isin(interactions.index.get_level_values('item_id').unique())

        train_item_features = dataset.build_item_features(
                train_item_features.loc[known_items_mask]["features"].items()
        )

        self._train_item_features = train_item_features

        self._model.fit(train_mat, user_features=self._train_user_features, item_features=self._train_item_features, epochs=self.num_epoch,
        num_threads=self.num_threads)


    def recommend(self, user_ids: Iterable[int], n: int):
        local_user_ids = np.array(tuple(map(self.user_mapping.get, user_ids)), dtype=np.int32)

        num_uniq_users = len(local_user_ids)
        local_item_ids = list(self.item_mapping.values())
        num_uniq_items = len(local_item_ids)

        local_item_ids *= num_uniq_users
        local_user_ids = local_user_ids.repeat(num_uniq_items)

        predicted_scores = self._model.predict(
            local_user_ids,
            local_item_ids,
            item_features=self._train_item_features,
            user_features=self._train_user_features,
            num_threads=self.num_threads)

        predicted_scores = predicted_scores.reshape((num_uniq_users, num_uniq_items))

        final_item_ids = []

        for row_num, user_id in enumerate(user_ids):
            additional_N = len(self._known_items_per_user_id.get(user_id, []))
            total_N = n + additional_N
            top_item_local_ids = np.argpartition(predicted_scores[row_num], -np.arange(total_N))[-total_N:][::-1]

            final_recs = list(map(self.inv_item_mapping.get, top_item_local_ids))

            if additional_N > 0:
                filter_items = self._known_items_per_user_id[user_id]
                final_recs = [item for item in final_recs if item not in filter_items][:n]

            final_item_ids.extend(final_recs)

        recs = pd.DataFrame(
            {
                "user_id": np.asarray(tuple(user_ids)).repeat(n),
                "item_id": final_item_ids
            }
        )

        recs["rank"] = recs.groupby("user_id").cumcount() + 1

        return recs.set_index(["user_id", "item_id"])
