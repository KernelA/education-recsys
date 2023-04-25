from typing import Iterable

import numpy as np
import pandas as pd
from implicit.recommender_base import RecommenderBase
from scipy.sparse import csr_matrix


class ImplicitRecommender:
    def __init__(self, model: RecommenderBase, user_mapping, item_mapping, inv_item_mapping):
        self.model = model
        self.user_mapping = user_mapping
        self.inv_item_mapping = inv_item_mapping
        self._train_matrix = None
        self.item_mapping = item_mapping

    def model_name(self):
        return self.model.__class__.__name__

    def fit(self, train_matrix: csr_matrix, progress: bool = True):
        self._train_matrix = train_matrix
        self.model.fit(train_matrix, show_progress=progress)

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
