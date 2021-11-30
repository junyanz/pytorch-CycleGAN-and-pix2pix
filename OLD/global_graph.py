from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from l5kit.data import PERCEPTION_LABEL_TO_INDEX


class VectorizedEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        """A module which associates learnable embeddings to types

        :param embedding_dim: features of the embedding
        :type embedding_dim: int
        """
        super(VectorizedEmbedding, self).__init__()
        # Torchscript did not like enums, so we are going more primitive.
        self.polyline_types = {
            "AGENT_OF_INTEREST": 0,
            "AGENT_NO": 1,
            "AGENT_CAR": 2,
            "AGENT_BIKE": 3,
            "AGENT_PEDESTRIAN": 4,
            "TL_UNKNOWN": 5,  # unknown TL state for lane
            "TL_RED": 6,
            "TL_YELLOW": 7,
            "TL_GREEN": 8,
            "TL_NONE": 9,  # no TL for lane
            "CROSSWALK": 10,
            "LANE_BDRY_LEFT": 11,
            "LANE_BDRY_RIGHT": 12,
        }

        self.embedding = nn.Embedding(len(self.polyline_types), embedding_dim)

        # Torch script did not like dicts as Tensor selectors, so we are going more primitive.
        self.PERCEPTION_LABEL_CAR: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        self.PERCEPTION_LABEL_PEDESTRIAN: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_PEDESTRIAN"]
        self.PERCEPTION_LABEL_CYCLIST: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CYCLIST"]

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Model forward: embed the given elements based on their type.

        Assumptions:
        - agent of interest is the first one in the batch
        - other agents follow
        - then we have polylines (lanes)
        """

        with torch.no_grad():
            polyline_types = data_batch["type"]
            other_agents_types = data_batch["all_other_agents_types"]

            other_agents_len = other_agents_types.shape[1]
            lanes_len = data_batch["lanes_mid"].shape[1]
            crosswalks_len = data_batch["crosswalks"].shape[1]
            lanes_bdry_len = data_batch["lanes"].shape[1]
            total_len = 1 + other_agents_len + lanes_len + crosswalks_len + lanes_bdry_len

            other_agents_start_idx = 1
            lanes_start_idx = other_agents_start_idx + other_agents_len
            crosswalks_start_idx = lanes_start_idx + lanes_len
            lanes_bdry_start_idx = crosswalks_start_idx + crosswalks_len

            indices = torch.full(
                (len(polyline_types), total_len),
                fill_value=self.polyline_types["AGENT_NO"],
                dtype=torch.long,
                device=polyline_types.device,
            )

            # set agent of interest
            indices[:, 0].fill_(self.polyline_types["AGENT_OF_INTEREST"])
            # set others
            indices[:, other_agents_start_idx:lanes_start_idx][other_agents_types == self.PERCEPTION_LABEL_CAR].fill_(
                self.polyline_types["AGENT_CAR"]
            )
            indices[:, other_agents_start_idx:lanes_start_idx][
                other_agents_types == self.PERCEPTION_LABEL_PEDESTRIAN
            ].fill_(self.polyline_types["AGENT_PEDESTRIAN"])
            indices[:, other_agents_start_idx:lanes_start_idx][
                other_agents_types == self.PERCEPTION_LABEL_CYCLIST
            ].fill_(self.polyline_types["AGENT_BIKE"])

            # set lanes given their TL state.
            indices[:, lanes_start_idx:crosswalks_start_idx].copy_(data_batch["lanes_mid"][:, :, 0, -1]).add_(
                self.polyline_types["TL_UNKNOWN"]
            )

            indices[:, crosswalks_start_idx:lanes_bdry_start_idx].fill_(self.polyline_types["CROSSWALK"])
            indices[:, lanes_bdry_start_idx::2].fill_(self.polyline_types["LANE_BDRY_LEFT"])
            indices[:, lanes_bdry_start_idx + 1:: 2].fill_(self.polyline_types["LANE_BDRY_RIGHT"])

        return self.embedding.forward(indices)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiheadAttentionGlobalHead(nn.Module):
    """Global graph making use of multi-head attention.
    """

    def __init__(self, d_model: int, num_timesteps: int, num_outputs: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs
        self.encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.output_embed = MLP(d_model, d_model * 4, num_timesteps * num_outputs, num_layers=3)

    def forward(
        self, inputs: torch.Tensor, type_embedding: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Model forward:

        :param inputs: model inputs
        :param type_embedding: type embedding describing the different input types
        :param mask: availability mask

        :return tuple of outputs, attention
        """
        # dot-product attention:
        #   - query is ego's vector
        #   - key is inputs plus type embedding
        #   - value is inputs
        out, attns = self.encoder(inputs[[0]], inputs + type_embedding, inputs, mask)
        outputs = self.output_embed(out[0]).view(-1, self.num_timesteps, self.num_outputs)
        return outputs, attns
