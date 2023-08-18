import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .model import NeuralNetRecommender


@torch.no_grad()
def valid_one_epoch(*,
                    model: NeuralNetRecommender,
                    triplet_loss: torch.nn.TripletMarginLoss,
                    data_loader: DataLoader,
                    device: torch.device):

    model.eval()

    loss_value = 0
    num_samples = 0

    for batch in data_loader:
        batch_samples = batch["local_user_id"].shape[0]
        user_features, pos_item_features, neg_item_features = model(
            batch["local_user_id"].to(device),
            batch["user_features"].to(device),
            batch["local_pos_item_id"].to(device),
            batch["pos_item_features"].to(device),
            batch["local_neg_item_id"].to(device),
            batch["neg_item_features"].to(device)
        )
        loss = triplet_loss(user_features, pos_item_features, neg_item_features)
        loss_value += batch_samples * loss.item()
        num_samples += batch_samples

    return loss_value / num_samples


def train_one_epoch(*,
                    model: NeuralNetRecommender,
                    triplet_loss: torch.nn.TripletMarginLoss,
                    optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader,
                    device: torch.device):
    model.train()

    loss_value = 0
    num_samples = 0

    for batch in data_loader:
        optimizer.zero_grad()

        batch_samples = batch["local_user_id"].shape[0]

        user_features, pos_item_features, neg_item_features = model(
            batch["local_user_id"].to(device),
            batch["user_features"].to(device),
            batch["local_pos_item_id"].to(device),
            batch["pos_item_features"].to(device),
            batch["local_neg_item_id"].to(device),
            batch["neg_item_features"].to(device)
        )

        loss = triplet_loss(user_features, pos_item_features, neg_item_features)
        loss.backward()
        optimizer.step()
        loss_value += batch_samples * loss.item()
        num_samples += batch_samples

    return loss_value / num_samples
