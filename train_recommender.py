import pathlib

import hydra
import polars as pl
import wandb
from omegaconf import OmegaConf

from recs_utils.load_data import MTSDataset
from recs_utils.metrics import ITEM_ID_COL, USER_ID_COL, model_cross_validate
from recs_utils.split import TimeRangeSplit


@hydra.main(config_path="configs", config_name="cross_val", version_base="1.3")
def main(config):
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    interactions_path = orig_cwd / config.data.dump_files.interactions_path
    users_path = orig_cwd / config.data.dump_files.users_path
    items_path = orig_cwd / config.data.dump_files.items_path

    user_item_interactions = pl.read_parquet(interactions_path)
    user_item_interactions = MTSDataset.filter_noise_interactions(user_item_interactions)
    users = pl.read_parquet(users_path)
    items = pl.read_parquet(items_path)

    def model_factory():
        return hydra.utils.instantiate(config.model)

    dt_col = config.cv.dt_column
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

    full_config = OmegaConf.to_object(config)

    wandb.init(project=config.wandb.project_name, config=full_config,
               group=config.wandb.group, reinit=True)
    folds_info_with_stats = pl.DataFrame([info for _, _, info in folds_with_stats])
    folds_info_with_stats.write_csv("cv_stat.csv")
    wandb.log({"cv_stat": wandb.Table(dataframe=folds_info_with_stats.to_pandas())})

    del folds_info_with_stats

    metrics, model = model_cross_validate(
        user_item_interactions,
        items,
        users,
        folds_with_stats,
        model_factory,
        config.cv.num_recs)

    wandb.log({"cv_results": wandb.Table(dataframe=metrics.to_pandas())})
    metrics.write_csv("cv_res.csv")

    mean_map_across_folds = metrics.filter(
        pl.col("name") == f"MAP@{config.cv.num_recs}").get_column("value").mean()

    wandb.log({f"mean_MAP@{config.cv.num_recs}": mean_map_across_folds})

    return mean_map_across_folds


if __name__ == "__main__":
    main()
