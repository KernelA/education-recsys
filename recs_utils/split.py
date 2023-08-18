import datetime
from typing import Optional, Sequence, Tuple, Union

import polars as pl

from .constants import TARGET_COL
from .neg_samples import compute_stat, get_pos_neg_samples


class TimeRangeSplit:
    def __init__(self,
                 *,
                 start_date: datetime.datetime,
                 interval: datetime.timedelta,
                 folds: int,
                 end_date: Optional[datetime.datetime] = None,
                 tz=None,
                 normalize: bool = False,
                 train_min_date: Optional[datetime.datetime] = None,
                 filter_cold_users: bool = True,
                 filter_cold_items: bool = True,
                 filter_already_seen: bool = True):

        self.start_date = start_date
        self.end_date = end_date
        self.tz = tz
        self.normalize = normalize
        self.train_min_date = train_min_date
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

        if self.end_date is None:
            self.end_date = start_date + (folds + 1) * interval

        self._date_col = "date_col"
        self.date_range = pl.date_range(
            start=start_date,
            end=self.end_date,
            interval=interval,
            time_zone=self.tz,
            closed="left",
            eager=True,
            name=self._date_col
        )

        self.max_n_splits = max(0, len(self.date_range) - 1)

        if self.max_n_splits == 0:
            raise ValueError("Provided parameters set an empty date range.")

    def _set_diff(self, left: pl.DataFrame, right: pl.DataFrame, col_names: Union[str, Sequence[str]]):
        """Same as np.setdiff1d left join + filtering
        """
        # Join with empty right may lead to error
        if len(right) == 0:
            return left.lazy().select(pl.col(col_names)).unique()

        return left.lazy().select(pl.col(col_names)).unique()\
            .join(
                right.lazy().select(
                    pl.col(col_names)
                ).unique(),
                how="anti",
                on=col_names
        ).select(pl.col(col_names).unique())

    def _filter_cold_entities(self,
                              train_index: pl.DataFrame,
                              test_index: pl.DataFrame,
                              test_mask: pl.Series,
                              interactions: pl.DataFrame,
                              entity_level_name: str):

        cold_ent_ids = self._set_diff(test_index, train_index, entity_level_name).collect()

        cold_end_index = interactions.lazy().select(pl.col(entity_level_name)).filter(test_mask).filter(pl.col(
            entity_level_name).is_in(cold_ent_ids.to_series())).collect()

        test_index_at_entity = self._set_diff(
            test_index, cold_end_index, entity_level_name).collect()

        test_index = test_index.filter(
            pl.col(entity_level_name).is_in(test_index_at_entity.to_series()))

        test_mask = interactions.select(
            pl.col(entity_level_name).is_in(
                test_index.select(pl.col(entity_level_name).unique()).to_series()
            )
        ).to_series()

        return test_index, test_mask, len(cold_ent_ids), len(cold_end_index)

    def split(self,
              interactions: pl.DataFrame,
              user_column: str = 'user_id',
              item_column: str = 'item_id',
              datetime_column: str = 'date',
              add_pos_neg_info: bool = False,
              fold_stats: bool = False):

        df_datetime = interactions.select(pl.col(datetime_column)).to_series()

        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.is_not_null()

        min_date = df_datetime.min()
        max_date = df_datetime.max()

        date_range = self.date_range.filter(
            (self.date_range >= min_date) & (self.date_range <= max_date))

        for start, end in zip(date_range, date_range[1:]):
            fold_info = {
                'Start date': start,
                'End date': end
            }
            train_mask = train_min_mask & (df_datetime < start)

            train_idx = interactions.filter(train_mask).select(
                pl.col(user_column),
                pl.col(item_column)
            )

            if fold_stats:
                fold_info['Train'] = len(train_idx)

                if add_pos_neg_info:
                    train_inter = interactions.join(
                        train_idx, on=[user_column, item_column])

                    train_user_stat = compute_stat(train_inter)
                    train_pos_samples, train_neg_samples = get_pos_neg_samples(
                        train_inter, train_user_stat)

                    fold_info["Train pos samples"] = len(train_pos_samples)
                    fold_info["Train neg samples"] = len(train_neg_samples)

                    train_idx = train_idx.lazy().join(
                        train_pos_samples.vstack(train_neg_samples).lazy(), on=[user_column, item_column]).select(
                            user_column, item_column, TARGET_COL
                    ).collect()

                    del train_neg_samples
                    del train_pos_samples

            test_mask = (df_datetime >= start) & (df_datetime < end)

            test_idx = interactions.filter(test_mask).select(
                pl.col(user_column),
                pl.col(item_column)
            )

            if self.filter_cold_users:
                test_idx, test_mask, num_cold_entities, num_cold_interactions = self._filter_cold_entities(
                    train_idx, test_idx, test_mask, interactions, user_column)

                if fold_stats:
                    fold_info['New users'] = num_cold_entities
                    fold_info['New users interactions'] = num_cold_interactions

            if self.filter_cold_items:
                test_idx, test_mask, num_cold_entities, num_cold_interactions = self._filter_cold_entities(
                    train_idx, test_idx, test_mask, interactions, item_column)

                if fold_stats:
                    fold_info['New items'] = num_cold_entities
                    fold_info['New items interactions'] = num_cold_interactions

            if self.filter_already_seen:
                intersection_idx = train_idx.join(
                    test_idx, on=[user_column, item_column], how="inner"
                )

                test_idx = self._set_diff(test_idx, intersection_idx, [
                    user_column, item_column]).collect()

                if fold_stats:
                    fold_info['Known interactions'] = len(intersection_idx)

            if fold_stats:
                fold_info['Test'] = len(test_idx)

                if add_pos_neg_info:
                    test_inter = interactions.join(test_idx, on=[user_column, item_column])
                    test_pos_samples, test_neg_samples = get_pos_neg_samples(
                        test_inter, train_user_stat)
                    fold_info["Test pos samples"] = len(test_pos_samples)
                    fold_info["Test neg samples"] = len(test_neg_samples)

                    test_idx = test_idx.lazy().join(
                        test_pos_samples.vstack(test_neg_samples).lazy(), on=[user_column, item_column]).select(
                            user_column, item_column, TARGET_COL
                    ).collect()

            yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df: pl.DataFrame, datetime_column: str = 'date'):
        df_datetime = df.select(pl.col(datetime_column)).to_series()

        if self.train_min_date is not None:
            df_datetime = df_datetime.filter(df_datetime >= self.train_min_date)

        min_date = df_datetime.min()
        max_date = df_datetime.max()

        date_range = self.date_range.filter(
            (self.date_range >= min_date) & (self.date_range <= max_date)
        )

        return max(0, len(date_range) - 1)


def train_test_split(data: pl.DataFrame, dates: Tuple[pl.Date, pl.Date]):
    train = data.filter(pl.col("start_date") < dates[0])
    test = data.filter((pl.col("start_date") >= dates[0]) & (pl.col("start_date") < dates[1]))

    return train, test
