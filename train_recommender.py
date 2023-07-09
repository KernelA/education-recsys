import pathlib

import polars as pl
import hydra

from recs_utils.split import TimeRangeSplit
from recs_utils.metrics import model_cross_validate, USER_ID_COL, ITEM_ID_COL


@hydra.main(config_path="configs", config_name="cross_val", version_base="1.3")
def main(config):
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    interactions_path = orig_cwd / config.dump_files.interactions_path
    users_path = orig_cwd / config.dump_files.users_path
    items_path = orig_cwd / config.dump_files.items_path

    user_item_interactions = pl.read_parquet(interactions_path)
    users = pl.read_parquet(users_path)
    items = pl.read_parquet(items_path)

    def model_factory():
        return hydra.utils.instantiate(config.model)

    dt_col = config.model.dt_column
    last_date = user_item_interactions.get_column(dt_col).max()
    folds = config.cv.num_periods
    interval = hydra.utils.instantiate(config.cv.period)
    start_date = last_date - interval * (folds + 1)
    cv = TimeRangeSplit(start_date=start_date, interval=interval, folds=folds)

    folds_with_stats = list(cv.split(
        user_item_interactions,
        user_column=USER_ID_COL,
        item_column=ITEM_ID_COL,
        datetime_column=dt_col,
        fold_stats=True
    )
    )

    folds_info_with_stats = pl.DataFrame([info for _, _, info in folds_with_stats])
    folds_info_with_stats.write_csv("cv_stat.csv")

    del folds_info_with_stats

    metrics, model = model_cross_validate(
        user_item_interactions,
        items,
        users,
        folds_with_stats,
        model_factory,
        config.cv.num_recs)

    metrics.write_csv("cv_res.csv")

    return metrics.filter(pl.col("name") == "MAP").get_column("value").mean()


if __name__ == "__main__":
    main()
