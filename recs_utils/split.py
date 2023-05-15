from typing import Tuple

import pandas as pd
import polars as pl


class TimeRangeSplit:
    def __init__(self,
                 start_date,
                 end_date=None,
                 freq: str = 'D',
                 periods=None,
                 tz=None,
                 normalize=False,
                 train_min_date=None,
                 filter_cold_users: bool = True,
                 filter_cold_items: bool = True,
                 filter_already_seen: bool = True):
        self.start_date = start_date

        if end_date is None and periods is None:
            raise ValueError(
                "Either 'end_date' or 'periods' must be non-zero, not both at the same time.")

        self.end_date = end_date
        self.freq = freq
        self.periods = periods
        self.tz = tz
        self.normalize = normalize
        self.train_min_date = pd.to_datetime(train_min_date, errors='raise')
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

        self.date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            periods=periods,
            tz=tz,
            normalize=normalize)

        self.max_n_splits = max(0, len(self.date_range) - 1)

        if self.max_n_splits == 0:
            raise ValueError("Provided parametrs set an empty date range.")

    def _filter_cold_entities(self,
                              train_index: pd.Index,
                              test_index: pd.Index,
                              test_mask,
                              interactions: pd.DataFrame,
                              entity_level_name: str):
        cold_ent_ids = test_index.get_level_values(entity_level_name).difference(
            train_index.get_level_values(entity_level_name))
        cold_end_index = interactions.index[test_mask & interactions.index.isin(
            cold_ent_ids, level=entity_level_name)]

        test_index_at_entity = test_index.get_level_values(entity_level_name).difference(
            cold_end_index.get_level_values(entity_level_name))

        test_index = test_index[test_index.isin(test_index_at_entity, level=entity_level_name)]
        test_mask = interactions.index.isin(test_index, level=entity_level_name)

        return test_index, test_mask, len(cold_ent_ids), len(cold_end_index)

    def split(self,
              df: pd.DataFrame,
              user_column: str = 'user_id',
              item_column: str = 'item_id',
              datetime_column: str = 'date',
              fold_stats: bool = False):

        df_datetime = df[datetime_column]

        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.notnull()

        date_range = self.date_range[(self.date_range >= df_datetime.min())
                                     & (self.date_range <= df_datetime.max())]

        for start, end in zip(date_range, date_range[1:]):
            fold_info = {
                'Start date': start,
                'End date': end
            }
            train_mask = train_min_mask & (df_datetime < start)
            train_idx: pd.Index = df.index[train_mask]

            if fold_stats:
                fold_info['Train'] = len(train_idx)

            test_mask = (df_datetime >= start) & (df_datetime < end)
            test_idx: pd.Index = df.index[test_mask]

            if self.filter_cold_users:
                test_idx, test_mask, num_cold_entities, num_cold_interactions = self._filter_cold_entities(
                    train_idx, test_idx, test_mask, df, user_column)

                if fold_stats:
                    fold_info['New users'] = num_cold_entities
                    fold_info['New users interactions'] = num_cold_interactions

            if self.filter_cold_items:
                test_idx, test_mask, num_cold_entities, num_cold_interactions = self._filter_cold_entities(
                    train_idx, test_idx, test_mask, df, item_column)

                if fold_stats:
                    fold_info['New items'] = num_cold_entities
                    fold_info['New items interactions'] = num_cold_interactions

            if self.filter_already_seen:
                intersection_idx = train_idx.intersection(test_idx)
                test_idx = test_idx[~test_idx.isin(intersection_idx)]

                if fold_stats:
                    fold_info['Known interactions'] = len(intersection_idx)

            if fold_stats:
                fold_info['Test'] = len(test_idx)

            yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df, datetime_column='date'):
        df_datetime = df[datetime_column]

        if self.train_min_date is not None:
            df_datetime = df_datetime[df_datetime >= self.train_min_date]

        date_range = self.date_range[(self.date_range >= df_datetime.min()) &
                                     (self.date_range <= df_datetime.max())]

        return max(0, len(date_range) - 1)


def train_test_split(data: pl.DataFrame, dates: Tuple[pl.Date, pl.Date]):
    train = data.filter(pl.col("start_date") < dates[0])
    test = data.filter((pl.col("start_date") >= dates[0]) & (pl.col("start_date") < dates[1]))

    return train, test
