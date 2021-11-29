'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from urban_driver.common import  build_target_normalization, pad_avail, pad_points
from urban_driver.global_graph import MultiheadAttentionGlobalHead, VectorizedEmbedding
from urban_driver.local_graph import LocalSubGraph, SinusoidalPositionalEmbedding
from urban_driver.global_graph_map import VectorizedMapEmbedding

class MapEncoder(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        self.type_embedding = VectorizedMapEmbedding(self._d_global)
        self._d_local = 256
        self._d_global = 256
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.register_buffer("other_agent_std", torch.tensor([33.2631, 21.3976, 1.5490]))

    def embed_polyline(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Embeds the inputs, generates the positional embedding and calls the local subgraph.

        :param features: input features
        :tensor features: [batch_size, num_elements, max_num_points, max_num_features]
        :param mask: availability mask
        :tensor mask: [batch_size, num_elements, max_num_points]

        :return tuple of local subgraphout output, (in-)availability mask
        """

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

    def model_call(self, static_polys: torch.Tensor, static_avail: torch.Tensor,
                   type_embedding: torch.Tensor, lane_bdry_len: int,) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Encapsulates calling the global_head (TODO?) and preparing needed data.

        :param static_polys: static elements - i.e. vectors corresponding to map elements
        :param static_avail: availability of map elements
        :param type_embedding:
        :param lane_bdry_len:
        """
        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]

        # Standardize inputs
        static_polys_feats = static_polys / self.other_agent_std

        all_polys = static_polys_feats
        all_avail = static_avail

        # Embed inputs, calculate positional embedding, call local subgraph
        all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        if hasattr(self, "global_from_local"):
            all_embs = self.global_from_local(all_embs)

        all_embs = F.normalize(all_embs, dim=-1) * (self._d_global ** 0.5)
        all_embs = all_embs.transpose(0, 1)

        invalid_polys[:, 0] = 0  # make AoI always available in global graph

        # call and return global graph
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys)
        return outputs, attns

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Load and prepare vectors for the model call

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

        # batch_size x (1 + M) x num features
        type_embedding = self.type_embedding(data_batch).transpose(0, 1)
        lane_bdry_len = data_batch["lanes"].shape[1]

        # call the model with these features
        outputs, attns = self.model_call(
            map_polys,  map_availabilities, type_embedding, lane_bdry_len
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
