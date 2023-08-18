import pathlib

import hydra
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from recs_utils.constants import ITEM_ID_COL, USER_ID_COL
from recs_utils.log_set import init_logging
from recs_utils.neural_network.dataset import TripletDataset
from recs_utils.neural_network.model import NeuralNetRecommender
from recs_utils.neural_network.training import train_one_epoch, valid_one_epoch
from recs_utils.utils import save_pickle


def save_model(model: torch.nn.Module, path_to_file: pathlib.Path):
    torch.save(model.state_dict(), path_to_file)


def simple_collat_fn(sample):
    return sample[0]


@hydra.main(config_path="configs", config_name="nn_rec", version_base="1.3")
def main(config):
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    data_dir = orig_cwd / config.data_dir

    train_inter = pl.read_parquet(data_dir / "interactions" / "train_inter.parquet")
    feature_dir = orig_cwd / config.feature_dir

    user_features = pl.read_parquet(feature_dir / "user_features.parquet")
    item_features = pl.read_parquet(feature_dir / "items.parquet")

    item_encoder = LabelEncoder()
    item_encoder.fit(train_inter.get_column(ITEM_ID_COL).unique().to_numpy())

    user_encoder = LabelEncoder()
    user_encoder.fit(train_inter.get_column(USER_ID_COL).unique().to_numpy())

    save_pickle("user_encoder.pickle", user_encoder)
    save_pickle("item_encoder.pickle", item_encoder)

    dataset = TripletDataset(train_inter, user_features, item_features,
                             user_encoder=user_encoder,
                             item_encoder=item_encoder)

    train_dataset, valid_dataset = data.random_split(
        dataset, [config.train_params.train_size, 1 - config.train_params.train_size])

    train_loader = data.DataLoader(
        train_dataset,
        sampler=data.BatchSampler(data.RandomSampler(train_dataset),
                                  batch_size=config.train_params.batch_size, drop_last=True),
        pin_memory=True,
        collate_fn=simple_collat_fn)

    valid_loader = data.DataLoader(
        valid_dataset,
        sampler=data.BatchSampler(data.RandomSampler(valid_dataset),
                                  batch_size=config.train_params.batch_size, drop_last=False),
        pin_memory=True,
        collate_fn=simple_collat_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: NeuralNetRecommender = hydra.utils.instantiate(config.model,
                                                          num_users=train_inter.get_column(
                                                              USER_ID_COL).n_unique(),
                                                          num_items=train_inter.get_column(
                                                              ITEM_ID_COL).n_unique(),
                                                          # minus user_id col
                                                          num_user_features=user_features.shape[1] - 1,
                                                          # minus item_id col
                                                          num_item_features=item_features.shape[1] - 1,
                                                          )
    model.to(device)
    model = torch.jit.script(model)

    loss_module = hydra.utils.instantiate(config.loss)
    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())
    scheduler = None

    if config.scheduler is not None:
        scheduler = hydra.utils.instantiate(config.scheduler, optimizer)

    last_valid_loss = None

    checkpoint_dir = pathlib.Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    with SummaryWriter(log_dir="training-logs", flush_secs=20) as logger:
        epoch_range = trange(config.train_params.epochs)

        for epoch in epoch_range:
            train_loss = train_one_epoch(
                model=model,
                triplet_loss=loss_module,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device
            )

            logger.add_scalar("Loss/train", train_loss, global_step=epoch, new_style=True)

            if epoch % 2 == 0:
                valid_loss = valid_one_epoch(
                    model=model,
                    triplet_loss=loss_module,
                    data_loader=valid_loader,
                    device=device
                )

                if last_valid_loss is None:
                    last_valid_loss = valid_loss
                    save_model(model, checkpoint_dir / "best-val-loss.ckpt")

                if valid_loss < last_valid_loss:
                    last_valid_loss = valid_loss
                    save_model(model, checkpoint_dir / "best-val-loss.ckpt")

                logger.add_scalar("Loss/valid", valid_loss, global_step=epoch, new_style=True)
                epoch_range.set_postfix({"Train epoch loss": train_loss,
                                        "Valid epoch loss": valid_loss})

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss)
                else:
                    scheduler.step()

            epoch_range.set_postfix({"Train epoch loss": train_loss})

            if epoch % 2 == 0 or epoch == config.train_params.epochs - 1:
                save_model(model, checkpoint_dir / "last.ckpt")


if __name__ == "__main__":
    init_logging(log_config="log_settings.yaml")
    main()
