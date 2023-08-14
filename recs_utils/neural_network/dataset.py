from typing import NamedTuple, Sequence

import numpy as np
import polars as pl
from torch.utils import data

from ..constants import ITEM_ID_COL, USER_ID_COL


class ShiftInfo:
    __slots__ = ["index", "items"]

    def __init__(self, items: Sequence[int]):
        self.index = 0
        self.items = items

    def next_item(self):
        item = self.items[self.index]
        self.index += 1
        self.index %= len(self.items)
        return item


class UserDataset(data.Dataset):
    def __init__(self, user_ids: pl.Series, user_features: pl.DataFrame) -> None:
        super().__init__()
        self.user_ids = user_ids
        self.user_features = user_features

    def __len__(self):
        return len(self.user_ids)

    def get_by_user_id(self, user_id: int):
        user_features = self.user_features.filter(pl.col(USER_ID_COL) == user_id).select(
            pl.col("*").exclude(USER_ID_COL)).to_numpy().reshape(-1).astype(np.float32)

        return {"user_id": user_id, "user_features": user_features}

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        return self.get_by_user_id(user_id)


class ItemDataset(data.Dataset):
    def __init__(self, item_ids: pl.Series, item_features: pl.DataFrame, item_key_name: str, item_features_key_name: str):
        super().__init__()
        self.item_ids = item_ids
        self.item_features = item_features
        self.item_key_name = item_key_name
        self.item_features_name = item_features_key_name

    def __len__(self):
        return len(self.item_ids)

    def get_by_item_id(self, item_id: int):
        item_features = self.item_features.filter(pl.col(ITEM_ID_COL) == item_id).select(
            pl.col("*").exclude(ITEM_ID_COL)).to_numpy().reshape(-1).astype(np.float32)
        return {self.item_key_name: item_id, self.item_features_name: item_features}

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        return self.get_by_item_id(item_id)


class TripletDataset(data.Dataset):
    def __init__(self, samples: pl.DataFrame, user_features: pl.DataFrame, item_features: pl.DataFrame):
        pos_samples = samples.filter(pl.col("target") == 1).select(
            pl.col(USER_ID_COL, ITEM_ID_COL))

        neg_samples = samples.filter(pl.col("target") == 0)

        self.user_dataset = UserDataset(
            pos_samples.get_column(USER_ID_COL).unique(), user_features)

        pos_item_ids = pos_samples.get_column(ITEM_ID_COL).unique()
        self.pos_items_dataset = ItemDataset(pos_item_ids, item_features.filter(pl.col(ITEM_ID_COL).is_in(pos_item_ids)),
                                             item_key_name="pos_item_id",
                                             item_features_key_name="pos_item_features")

        neg_item_ids = neg_samples.get_column(ITEM_ID_COL).unique()
        self.neg_item_dataset = ItemDataset(neg_item_ids, item_features.filter(pl.col(ITEM_ID_COL).is_in(neg_item_ids)), item_key_name="neg_item_id",
                                            item_features_key_name="neg_item_features")

        self.pos_samples = pos_samples.to_dicts()
        self.shifts = {}

        for row in neg_samples.groupby(USER_ID_COL).agg(pl.list(ITEM_ID_COL).flatten()).iter_rows(named=True):
            self.shifts[row[USER_ID_COL]] = ShiftInfo(row[ITEM_ID_COL])

    def __len__(self):
        return len(self.pos_samples)

    def __getitem__(self, index):
        info = self.pos_samples[index]
        user_id = info[USER_ID_COL]
        pos_item_id = info[ITEM_ID_COL]
        neg_item_id = self.shifts[user_id].next_item()

        user_info = self.user_dataset.get_by_user_id(user_id)
        pos_info = self.pos_items_dataset.get_by_item_id(pos_item_id)
        neg_info = self.neg_item_dataset.get_by_item_id(neg_item_id)

        return {**user_info, **pos_info, **neg_info}
