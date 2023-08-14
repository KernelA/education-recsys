import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
        for key in batch:
            batch[key] = batch[key].to(device)
        user_features, pos_item_features, neg_item_features = model(
            batch["user_id"],
            batch["user_features"],
            batch["pos_item_id"],
            batch["pos_item_features"],
            batch["neg_item_id"],
            batch["neg_item_features"]
        )
        loss = triplet_loss(user_features, pos_item_features, neg_item_features)
        loss_value += batch.shape[0] * loss.item()
        num_samples += batch.shape[0]

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
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        user_features, pos_item_features, neg_item_features = model(
            batch["user_id"],
            batch["user_features"],
            batch["pos_item_id"],
            batch["pos_item_features"],
            batch["neg_item_id"],
            batch["neg_item_features"]
        )

        loss = triplet_loss(user_features, pos_item_features, neg_item_features)
        loss.backward()
        optimizer.step()
        loss_value += batch.shape[0] * loss.item()
        num_samples += batch.shape[0]

    return loss_value / num_samples
