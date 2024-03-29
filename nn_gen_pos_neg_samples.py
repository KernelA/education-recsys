import datetime
import logging
import pathlib

import hydra
import polars as pl

from recs_utils.base_model import BaseRecommender
from recs_utils.constants import ITEM_ID_COL, TARGET_COL, USER_ID_COL
from recs_utils.load_data import MTSDataset
from recs_utils.log_set import init_logging
from recs_utils.neg_samples import select_pos_samples
from recs_utils.split import TimeRangeSplit


@hydra.main(config_path="configs", config_name="gen_samples", version_base="1.3")
def main(config):
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())

    users = pl.read_parquet(orig_cwd / config.data.dump_files.users_path)

    users = users.with_columns(
        pl.col("sex").fill_null("unknown"),
        pl.col("age").fill_null("unknown"))

    items = MTSDataset.select_genres(pl.read_parquet(
        orig_cwd / config.data.dump_files.items_path), cov=0.95, null_value="unknown")

    interactions = pl.read_parquet(orig_cwd / config.data.dump_files.interactions_path).filter(
        pl.col(USER_ID_COL).is_in(users.get_column(USER_ID_COL).unique()) & pl.col(
            ITEM_ID_COL).is_in(items.get_column(ITEM_ID_COL).unique())
    )

    interactions = MTSDataset.filter_noise_interactions(interactions)

    dt_col = config.cv.dt_column
    last_date = interactions.get_column(dt_col).max()
    folds = config.cv.num_periods
    assert folds == 1, "Only one fold supported"

    interval = hydra.utils.instantiate(config.cv.period)
    start_date = last_date - interval * (folds + 1)
    cv = TimeRangeSplit(start_date=start_date, interval=interval, folds=folds)

    train_index, test_index, fold_info = list(cv.split(
        interactions,
        user_column=USER_ID_COL,
        item_column=ITEM_ID_COL,
        datetime_column=dt_col,
        fold_stats=True,
        add_pos_neg_info=True
    )
    )[0]

    folds_info_with_stats = pl.DataFrame(fold_info)
    folds_info_with_stats.write_csv("cv_stat.csv")

    train_inter = interactions.join(train_index, on=[USER_ID_COL, ITEM_ID_COL], how="inner")
    test_inter = interactions.join(test_index, on=[USER_ID_COL, ITEM_ID_COL], how="inner")

    test_inter = select_pos_samples(test_inter)
    train_pos_inter = select_pos_samples(train_inter)
    train_neg_inter = train_inter.join(train_pos_inter.select(
        pl.col(USER_ID_COL), pl.col(ITEM_ID_COL)), on=[USER_ID_COL, ITEM_ID_COL], how="anti")

    assert train_pos_inter.get_column(USER_ID_COL).n_unique() == train_inter.get_column(
        USER_ID_COL).n_unique(), "Not all users have pos samples"

    ratio_users_with_neg_samples = train_neg_inter.get_column(
        USER_ID_COL).n_unique() / train_inter.get_column(USER_ID_COL).n_unique()

    logging.info(
        f"Ratio of negative samples for all train users: {ratio_users_with_neg_samples:.2%}")

    if train_neg_inter.get_column(
            USER_ID_COL).n_unique() != train_inter.get_column(USER_ID_COL).n_unique():
        logging.info("Sample additions negative samples per user")

        users_without_neg_samples = train_pos_inter.join(
            train_neg_inter, on=[USER_ID_COL], how="anti").get_column(USER_ID_COL).unique()

        simple_model: BaseRecommender = hydra.utils.instantiate(config.simple_model)
        simple_model.fit(train_inter, user_features=users, item_features=items)

        predicted_recs: pl.DataFrame = simple_model.recommend(train_inter.filter(
            pl.col(USER_ID_COL).is_in(users_without_neg_samples)), num_recs_per_user=config.cv.num_recs)

        train_neg_inter = train_neg_inter.select(pl.col(USER_ID_COL, ITEM_ID_COL, TARGET_COL)).vstack(
            predicted_recs.join(train_inter, on=[USER_ID_COL, ITEM_ID_COL], how="anti").select(
                pl.col(USER_ID_COL, ITEM_ID_COL)).with_columns(pl.lit(0, dtype=train_pos_inter.schema[TARGET_COL]).alias(TARGET_COL))
        )

    assert train_neg_inter.get_column(USER_ID_COL).n_unique() == train_inter.get_column(
        USER_ID_COL).n_unique(), "All users must have negative samples"

    logging.info("Mean number of neg samples per user %.2f", train_neg_inter.groupby(
        USER_ID_COL).agg(pl.n_unique(ITEM_ID_COL)).mean()[0, 1])

    train_samples = train_pos_inter.select(
        pl.col(USER_ID_COL, ITEM_ID_COL, TARGET_COL)).vstack(train_neg_inter)

    data_dir = pathlib.Path("interactions")
    data_dir.mkdir(exist_ok=True)

    train_samples.write_parquet(data_dir / "train_inter.parquet")
    test_inter.write_parquet(data_dir / "test_inter.parquet")

    users = users.filter(pl.col(USER_ID_COL).is_in(train_samples.get_column(USER_ID_COL).unique().append(
        test_inter.get_column(USER_ID_COL).unique())))

    data_dir = pathlib.Path("features")
    data_dir.mkdir(exist_ok=True)

    users.write_parquet(data_dir / "users.parquet")

    items = items.filter(
        pl.col(ITEM_ID_COL).is_in(train_samples.get_column(ITEM_ID_COL).unique().append(
            test_inter.get_column(ITEM_ID_COL).unique())))

    items.write_parquet(data_dir / "items.parquet")


if __name__ == "__main__":
    init_logging(log_config="log_settings.yaml")
    main()
