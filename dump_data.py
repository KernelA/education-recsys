import pathlib

import hydra

from recs_utils.load_data import MTSDataset


@hydra.main(config_path="configs", config_name="dump", version_base="1.3")
def main(config):
    raw_dir = pathlib.Path(config.data.raw_dir)
    out_dir = pathlib.Path(config.data.dump_files.interactions_path).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    for in_name, out_path, load_func in zip(
        ("interactions.csv", "items.csv", "users.csv"),
        (config.data.dump_files.interactions_path,
         config.data.dump_files.items_path, config.data.dump_files.users_path),
            (MTSDataset.load_interactions, MTSDataset.load_items, MTSDataset.load_users)
    ):
        data = load_func(raw_dir / in_name)
        data.write_parquet(out_path)


if __name__ == "__main__":
    main()
