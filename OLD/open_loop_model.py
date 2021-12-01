from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import build_target_normalization, pad_avail, pad_points
from .global_graph import MultiheadAttentionGlobalHead, VectorizedEmbedding
from .local_graph import LocalSubGraph, SinusoidalPositionalEmbedding


class VectorizedModel(nn.Module):
    """ Vectorized planning model.
    """

    def __init__(
        self,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        global_head_dropout: float,
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
    ) -> None:
        """ Initializes the model.

        :param history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param num_targets: number of values to predict
        :param weights_scaling: target weights for loss calculation
        :param global_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it
        :param criterion: loss function to use
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        """
        super().__init__()
        self.disable_map = disable_map
        self.disable_other_agents = disable_other_agents
        self.disable_lane_boundaries = disable_lane_boundaries

        self._history_num_frames_ego = history_num_frames_ego
        self._history_num_frames_agents = history_num_frames_agents
        self._num_targets = num_targets

        self._global_head_dropout = global_head_dropout

        self._d_local = 256
        self._d_global = 256

        self._agent_features = ["start_x", "start_y", "yaw"]
        self._lane_features = ["start_x", "start_y", "tl_feature"]
        self._vector_agent_length = len(self._agent_features)
        self._vector_lane_length = len(self._lane_features)
        self._subgraph_layers = 3

        self.register_buffer("weights_scaling", torch.as_tensor(weights_scaling))
        self.criterion = criterion

        self.normalize_targets = True
        num_outputs = len(weights_scaling)
        num_timesteps = num_targets // num_outputs

        if self.normalize_targets:
            scale = build_target_normalization(num_timesteps)
            self.register_buffer("xy_scale", scale)

        # normalization buffers
        self.register_buffer("agent_std", torch.tensor([1.6919, 0.0365, 0.0218]))
        self.register_buffer("other_agent_std", torch.tensor([33.2631, 21.3976, 1.5490]))

        self.input_embed = nn.Linear(self._vector_agent_length, self._d_local)
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.type_embedding = VectorizedEmbedding(self._d_global)

        self.disable_pos_encode = False

        self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)

        if self._d_global != self._d_local:
            self.global_from_local = nn.Linear(self._d_local, self._d_global)

        self.global_head = MultiheadAttentionGlobalHead(
            self._d_global, num_timesteps, num_outputs, dropout=self._global_head_dropout
        )

    def embed_polyline(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Embeds the inputs, generates the positional embedding and calls the local subgraph.

        :param features: input features
        :tensor features: [batch_size, num_elements, max_num_points, max_num_features]
        :param mask: availability mask
        :tensor mask: [batch_size, num_elements, max_num_points]

        :return tuple of local subgraphout output, (in-)availability mask
        """
        # embed inputs
        # [batch_size, num_elements, max_num_points, embed_dim]
        polys = self.input_embed(features)
        # calculate positional embedding
        # [1, 1, max_num_points, embed_dim]
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)
        # [batch_size, num_elements, max_num_points]
        invalid_mask = ~mask
        invalid_polys = invalid_mask.all(-1)
        # input features to local subgraph and return result -
        # local subgraph reduces features over elements, i.e. creates one descriptor
        # per element
        # [batch_size, num_elements, embed_dim]
        polys = self.local_subgraph(polys, invalid_mask, pos_embedding)
        return polys, invalid_polys

    def model_call(
        self,
        agents_polys: torch.Tensor,
        static_polys: torch.Tensor,
        agents_avail: torch.Tensor,
        static_avail: torch.Tensor,
        type_embedding: torch.Tensor,
        lane_bdry_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Encapsulates calling the global_head (TODO?) and preparing needed data.

        :param agents_polys: dynamic elements - i.e. vectors corresponding to agents
        :param static_polys: static elements - i.e. vectors corresponding to map elements
        :param agents_avail: availability of agents
        :param static_avail: availability of map elements
        :param type_embedding:
        :param lane_bdry_len:
        """
        # Standardize inputs
        agents_polys_feats = torch.cat(
            [agents_polys[:, :1] / self.agent_std, agents_polys[:, 1:] / self.other_agent_std], dim=1
        )
        static_polys_feats = static_polys / self.other_agent_std

        all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
        all_avail = torch.cat([agents_avail, static_avail], dim=1)

        # Embed inputs, calculate positional embedding, call local subgraph
        all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        if hasattr(self, "global_from_local"):
            all_embs = self.global_from_local(all_embs)

        all_embs = F.normalize(all_embs, dim=-1) * (self._d_global ** 0.5)
        all_embs = all_embs.transpose(0, 1)

        other_agents_len = agents_polys.shape[1] - 1

        # disable certain elements on demand
        if self.disable_other_agents:
            invalid_polys[:, 1: (1 + other_agents_len)] = 1  # agents won't create attention

        if self.disable_map:  # lanes (mid), crosswalks, and lanes boundaries.
            invalid_polys[:, (1 + other_agents_len):] = 1  # lanes won't create attention

        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]

        invalid_polys[:, 0] = 0  # make AoI always available in global graph

        # call and return global graph
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        return outputs, attns

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Load and prepare vectors for the model call, split into map and agents

        # ==== LANES ====
        # batch size x num lanes x num vectors x num features
        polyline_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            polyline_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in polyline_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in polyline_keys])

        map_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in polyline_keys], dim=1)
        map_polys[..., -1].fill_(0)
        # batch size x num lanes x num vectors
        map_availabilities = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)

        # ==== AGENTS ====
        # batch_size x (1 + M) x seq len x self._vector_length
        agents_polys = torch.cat(
            [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
        )
        # batch_size x (1 + M) x num vectors x self._vector_length
        agents_polys = pad_points(agents_polys, max_num_vectors)

        # batch_size x (1 + M) x seq len
        agents_availabilities = torch.cat(
            [
                data_batch["agent_polyline_availability"].unsqueeze(1),
                data_batch["other_agents_polyline_availability"],
            ],
            dim=1,
        )
        # batch_size x (1 + M) x num vectors
        agents_availabilities = pad_avail(agents_availabilities, max_num_vectors)

        # batch_size x (1 + M) x num features
        type_embedding = self.type_embedding(data_batch).transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        # call the model with these features
        outputs, attns = self.model_call(
            agents_polys, map_polys, agents_availabilities, map_availabilities, type_embedding, lane_bdry_len
        )

        # calculate loss or return predicted position for inference
        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            xy = data_batch["target_positions"]
            yaw = data_batch["target_yaws"]
            if self.normalize_targets:
                xy /= self.xy_scale
            targets = torch.cat((xy, yaw), dim=-1)
            target_weights = data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            pred_positions, pred_yaws = outputs[..., :2], outputs[..., 2:3]
            if self.normalize_targets:
                pred_positions *= self.xy_scale

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            if attns is not None:
                eval_dict["attention_weights"] = attns
            return eval_dict
