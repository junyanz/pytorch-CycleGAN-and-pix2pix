from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F



class VectorizedMapEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        """A module which associates learnable embeddings to types

        :param embedding_dim: features of the embedding
        :type embedding_dim: int
        """
        super(VectorizedMapEmbedding, self).__init__()
        # Torchscript did not like enums, so we are going more primitive.
        super().__init__()
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

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Model forward: embed the given elements based on their type.

        Assumptions:
        - then we have polylines (lanes)
        """

        with torch.no_grad():
            polyline_types = data_batch["type"]

            lanes_len = data_batch["lanes_mid"].shape[1]
            crosswalks_len = data_batch["crosswalks"].shape[1]
            lanes_bdry_len = data_batch["lanes"].shape[1]
            total_len = 1 + lanes_len + crosswalks_len + lanes_bdry_len

            crosswalks_start_idx = 1 + lanes_len
            lanes_bdry_start_idx = crosswalks_start_idx + crosswalks_len

            indices = torch.full(
                (len(polyline_types), total_len),
                fill_value=self.polyline_types["AGENT_NO"],
                dtype=torch.long,
                device=polyline_types.device,
            )
            # set lanes given their TL state.
            indices[:, :crosswalks_start_idx].copy_(data_batch["lanes_mid"][:, :, 0, -1]).add_(
                self.polyline_types["TL_UNKNOWN"]
            )

            indices[:, crosswalks_start_idx:lanes_bdry_start_idx].fill_(self.polyline_types["CROSSWALK"])
            indices[:, lanes_bdry_start_idx::2].fill_(self.polyline_types["LANE_BDRY_LEFT"])
            indices[:, lanes_bdry_start_idx + 1:: 2].fill_(self.polyline_types["LANE_BDRY_RIGHT"])

        return self.embedding.forward(indices)

