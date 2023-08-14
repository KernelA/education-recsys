import torch
from torch import nn


class EntityModel(nn.Module):
    def __init__(self,
                 *,
                 embedding_size: int,
                 num_entities: int,
                 num_features: int,
                 hidden_dim: int,
                 out_features: int) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)

        self.init_feature_mapping = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.feature_mapping = nn.Sequential(
            nn.Linear(embedding_size + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features, bias=False)
        )

    def forward(self, entity_ids: torch.LongTensor, features: torch.Tensor):
        assert entity_ids.shape[0] == features.shape[0]
        entity_embeddings = self.entity_embedding(entity_ids).view(entity_ids.shape[0], -1)

        features = self.init_feature_mapping(features)
        return self.feature_mapping(torch.cat((entity_embeddings, features), dim=1))


class NeuralNetRecommender(nn.Module):
    def __init__(self,
                 *,
                 embedding_size: int,
                 num_users: int,
                 num_items: int,
                 num_user_features: int,
                 num_item_features: int,
                 hidden_dim: int,
                 out_features: int) -> None:
        super().__init__()
        self.user_model = EntityModel(embedding_size=embedding_size,
                                      num_entities=num_users,
                                      num_features=num_user_features,
                                      hidden_dim=hidden_dim,
                                      out_features=out_features)
        self.item_model = EntityModel(embedding_size=embedding_size,
                                      num_entities=num_items,
                                      num_features=num_item_features,
                                      hidden_dim=hidden_dim,
                                      out_features=out_features)

    def forward(self,
                user_ids: torch.LongTensor,
                user_features: torch.Tensor,
                pos_item_ids: torch.LongTensor,
                pos_item_features: torch.Tensor,
                neg_item_ids: torch.LongTensor,
                neg_item_features: torch.Tensor):
        assert user_ids.shape[0] == neg_item_ids.shape[0] == pos_item_ids.shape[0], "Number of samples must be same as number of users"

        return self.user_model(user_ids, user_features), self.item_model(pos_item_ids, pos_item_features), self.item_model(neg_item_ids, neg_item_features)
