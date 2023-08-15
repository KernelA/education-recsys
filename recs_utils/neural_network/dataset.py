from typing import Iterator, Sequence, Union

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
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
    def __init__(self, user_features: pl.DataFrame, user_encoder: LabelEncoder) -> None:
        super().__init__()
        user_features = user_features.sort(USER_ID_COL)
        self.user_ids = user_features.get_column(USER_ID_COL).to_numpy()
        self.user_encoder = user_encoder
        self.user_features = torch.from_numpy(user_features.select(
            pl.col("*").exclude(USER_ID_COL)).to_numpy().astype(np.float32))

    def __len__(self):
        return len(self.user_ids)

    def get_by_user_id(self, user_id: Union[int, np.ndarray]):
        if isinstance(user_id, int):
            user_id = [user_id]

        indices = np.searchsorted(self.user_ids, user_id)

        return {
            "local_user_id": torch.from_numpy(self.user_encoder.transform(user_id)),
            "user_features": self.user_features[indices, :]
        }

    def __getitem__(self, index: Union[int, np.ndarray]):
        user_id = self.user_ids[index]
        info = self.get_by_user_id(user_id)
        return info


class ItemDataset(data.Dataset):
    def __init__(self,
                 item_features: pl.DataFrame,
                 item_encoder: LabelEncoder,
                 item_key_name: str, item_features_key_name: str):
        super().__init__()
        item_features = item_features.sort(ITEM_ID_COL)
        self.item_ids = item_features.get_column(ITEM_ID_COL).to_numpy()
        self.item_encoder = item_encoder
        self.item_features = torch.from_numpy(item_features.select(
            pl.col("*").exclude(ITEM_ID_COL)).to_numpy().astype(np.float32))
        self.item_key_name = item_key_name
        self.item_features_name = item_features_key_name

    def __len__(self):
        return len(self.item_ids)

    def get_by_item_id(self, item_id: Union[int, np.ndarray]):
        if isinstance(item_id, int):
            item_id = [item_id]

        indices = np.searchsorted(self.item_ids, item_id)

        return {
            self.item_key_name: torch.from_numpy(self.item_encoder.transform(item_id)),
            self.item_features_name: self.item_features[indices, :]
        }

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        return self.get_by_item_id(item_id)


class TripletDataset(data.Dataset):
    def __init__(self,
                 samples: pl.DataFrame,
                 user_features: pl.DataFrame,
                 item_features: pl.DataFrame,
                 user_encoder: LabelEncoder,
                 item_encoder: LabelEncoder):

        self.pos_samples = samples.filter(pl.col("target") == 1).select(
            pl.col(USER_ID_COL, ITEM_ID_COL))
        neg_samples = samples.filter(pl.col("target") == 0).select(pl.col(USER_ID_COL, ITEM_ID_COL))

        self.user_dataset = UserDataset(user_features.join(
            samples.select(pl.col(USER_ID_COL)), on=[USER_ID_COL], how="inner"), user_encoder=user_encoder)

        self.pos_items_dataset = ItemDataset(item_features.join(self.pos_samples.select(pl.col(ITEM_ID_COL)), on=ITEM_ID_COL, how="inner"),
                                             item_key_name="local_pos_item_id",
                                             item_features_key_name="pos_item_features",
                                             item_encoder=item_encoder)

        self.neg_item_dataset = ItemDataset(item_features.join(neg_samples.select(pl.col(ITEM_ID_COL)), on=ITEM_ID_COL, how="inner"),
                                            item_key_name="local_neg_item_id",
                                            item_features_key_name="neg_item_features",
                                            item_encoder=item_encoder)

        self.shifts = neg_samples.sort(USER_ID_COL).set_sorted(USER_ID_COL).with_columns(pl.count(ITEM_ID_COL).over(
            USER_ID_COL).alias("total_items")).with_columns(
            pl.lit(0).alias("start_index"),
            pl.col(ITEM_ID_COL).rank("ordinal").alias("pos").over(USER_ID_COL) - 1)

    def __len__(self):
        return len(self.pos_samples)

    def __getitem__(self, index: Union[int, np.ndarray]):
        info = self.pos_samples[index]

        user_ids = info.get_column(USER_ID_COL).to_numpy()
        pos_item_id = info.get_column(ITEM_ID_COL).to_numpy()

        selected_neg_items = self.shifts.filter(pl.col(USER_ID_COL).is_in(user_ids))

        neg_item_id = info.select(pl.col(USER_ID_COL)).join(
            selected_neg_items.filter(pl.col("pos") == pl.col("start_index")).select(
                pl.col(USER_ID_COL, ITEM_ID_COL),
            ), on=USER_ID_COL, how="inner").get_column(ITEM_ID_COL).to_numpy()

        assert len(user_ids) == len(pos_item_id)
        assert len(neg_item_id) == len(pos_item_id)

        selected_neg_items = selected_neg_items.with_columns(((pl.col("start_index") + 1) %
                                                              pl.col("total_items")).alias("start_index").cast(pl.Int32))

        self.shifts = self.shifts.join(
            selected_neg_items, on=[USER_ID_COL, ITEM_ID_COL], how="anti").vstack(selected_neg_items)

        user_info = self.user_dataset.get_by_user_id(user_ids)
        pos_info = self.pos_items_dataset.get_by_item_id(pos_item_id)
        neg_info = self.neg_item_dataset.get_by_item_id(neg_item_id)

        return {**user_info, **pos_info, **neg_info}
