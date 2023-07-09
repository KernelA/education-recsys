import pathlib

import hydra

from recs_utils.load_data import MTSDataset


@hydra.main(config_path="configs", config_name="dump", version_base="1.3")
def main(config):
    raw_dir = pathlib.Path(config.raw_dir)
    path_to_file = raw_dir / "interactions.csv"
    out_dir = pathlib.Path(config.dump_files.interactions).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    interactions = MTSDataset.load_interactions(path_to_file)
    interactions.write_parquet(config.dump_files.interactions)
    del interactions

    path_to_file = raw_dir / "items.csv"
    items = MTSDataset.load_items(path_to_file)
    items.write_parquet(config.dump_files.items_path)
    del items

    path_to_file = raw_dir / "users.csv"
    users = MTSDataset.load_users(path_to_file)
    users.write_parquet(config.dump_files.users)
    del users


if __name__ == "__main__":
    main()
