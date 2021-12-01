
from typing import Dict, List, Tuple

import torch
from torch import nn

from .common import pad_avail, pad_points, transform_points
from .open_loop_model import VectorizedModel


class VectorizedUnrollModel(VectorizedModel):
    """ Vectorized closed-loop planning model.
    """

    def __init__(
        self,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,  # criterion is only needed for training and not for evaluation
        global_head_dropout: float,
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
        detach_unroll: bool,
        warmup_num_frames: int,
        discount_factor: float,
        limit_predicted_yaw: bool = True,
    ) -> None:
        """ Initializes the model.

        :param history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param num_targets: number of values to predict
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param gobal_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        :param detach_unroll: detach gradient between steps (disable BPTT)
        :param warmup_num_frames: "sample" warmup_num_frames by following the model's policy
        :param discount_factor: discount future_timesteps via discount_factor**t
        :param limit_predicted_yaw: limit predicted yaw to 0.3 * tanh(x) if enabled - recommended for more stable
            training
        """

        num_targets = 3  # this will limit queries number to 1

        super().__init__(
            history_num_frames_ego,
            history_num_frames_agents,
            num_targets,
            weights_scaling,
            criterion,
            global_head_dropout,
            disable_other_agents,
            disable_map,
            disable_lane_boundaries,
        )

        self.detach_unroll = detach_unroll
        self.warmup_num_frames = warmup_num_frames
        self.discount_factor = discount_factor
        self.limit_predicted_yaw = limit_predicted_yaw

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ==== get additional info from the batch, or fall back to sensible defaults
        future_num_frames = data_batch["target_availabilities"].shape[1]

        # ==== Past and Static info
        agents_past_polys = torch.cat(
            [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
        )
        agents_past_avail = torch.cat(
            [data_batch["agent_polyline_availability"].unsqueeze(1), data_batch["other_agents_polyline_availability"]],
            dim=1,
        )

        static_keys = ["lanes_mid", "crosswalks"]
        if not self.disable_lane_boundaries:
            static_keys += ["lanes"]
        avail_keys = [f"{k}_availabilities" for k in static_keys]

        max_num_vectors = max([data_batch[key].shape[-2] for key in static_keys])

        static_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in static_keys], dim=1)
        static_polys[..., -1] = 0  # NOTE: this is an hack
        static_avail = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)

        # ===== Future information
        agents_future_positions = torch.cat(
            [data_batch["target_positions"].unsqueeze(1), data_batch["all_other_agents_future_positions"]], dim=1
        )
        agents_future_yaws = torch.cat(
            [data_batch["target_yaws"].unsqueeze(1), data_batch["all_other_agents_future_yaws"]], dim=1
        )
        agents_future_avail = torch.cat(
            [data_batch["target_availabilities"].unsqueeze(1), data_batch["all_other_agents_future_availability"]],
            dim=1,
        )

        # concat XY and yaw to mimic past
        agents_future_polys = torch.cat([agents_future_positions, agents_future_yaws], dim=3)

        # Combine past and future agent information.
        # Future information is ordered [T+1, T+2, ...], past information [T, T-1, T-2, ...].
        # We thus flip past vectors and by concatenating get [..., T-2, T-1, T, T+1, T+2, ...].
        # Now, at each step T the current time window of interest simply is represented by the indices
        # T + agents_past_polys.shape[2] - window_size + 1: T + agents_past_polys.shape[2] + 1.
        # During the training loop, we will fetch this information, as well as static features,
        # which is all represented in the space of T = 0.
        # We then transform this into the space of T and feed this to the model.
        # Eventually, we shift our time window one step into the future.
        # See below for more information about used coordinate spaces.
        agents_polys = torch.cat([torch.flip(agents_past_polys, [2]), agents_future_polys], dim=2)
        agents_avail = torch.cat([torch.flip(agents_past_avail.contiguous(), [2]), agents_future_avail], dim=2)
        window_size = agents_past_polys.shape[2]
        current_timestep = agents_past_polys.shape[2] - 1

        outputs_ts = []  # buffer for predictions in local spaces
        gts_ts = []  # buffer for gts in local spaces
        outputs_t0 = []  # buffer for final prediction in t0 space (for eval only)
        attns = []

        batch_size = agents_polys.shape[0]
        lane_bdry_len = data_batch["lanes"].shape[1]

        type_embedding = self.type_embedding(data_batch).transpose(0, 1)

        one = torch.ones_like(data_batch["target_yaws"][:, 0])
        zero = torch.zeros_like(data_batch["target_yaws"][:, 0])

        # ====== Transformation between local spaces
        # NOTE: we use the standard convention A_from_B to indicate that a matrix/yaw/translation
        # converts a point from the B space into the A space
        # e.g. if pB = (1,0) and A_from_B = (-1, 1) then pA = (0, 1)
        # NOTE: we use the following convention for names:
        # t0 -> space at 0, i.e. the space we pull out of the data for which ego is in (0, 0) with no yaw
        # ts -> generic space at step t = s > 0 (predictions at t=s are in this space)
        # tsplus -> space at s+1 (proposal new ts, built from prediction at t=s)
        # A_from_B -> indicate a full 2x3 RT matrix from B to A
        # yaw_A_from_B -> indicate a yaw from B to A
        # tr_A_from_B -> indicate a translation (XY) from B to A
        # NOTE: matrices (and yaw) we need to keep updated while we loop:
        # t0_from_ts -> bring a point from the current space into the data one (e.g. for visualisation)
        # ts_from_t0 -> bring a point from data space into the current one (e.g. to compute loss
        t0_from_ts = torch.eye(3, device=one.device).unsqueeze(0).repeat(batch_size, 1, 1)
        ts_from_t0 = t0_from_ts.clone()
        yaw_t0_from_ts = zero
        yaw_ts_from_t0 = zero

        for idx in range(future_num_frames):
            # === STEP FORWARD ====
            # pick the right point in time
            agents_polys_step = torch.flip(
                agents_polys[:, :, current_timestep - window_size + 1: current_timestep + 1], [2]
            ).clone()
            agents_avail_step = torch.flip(
                agents_avail[:, :, current_timestep - window_size + 1: current_timestep + 1].contiguous(), [2]
            ).clone()
            # PAD
            agents_polys_step = pad_points(agents_polys_step, max_num_vectors)
            agents_avail_step = pad_avail(agents_avail_step, max_num_vectors)

            # crop agents history accordingly
            # NOTE: before padding, agent_polys_step has a number of elements equal to:
            # max_history_num_frames + 1 (where the +1 comes from T0, which is the 0-th element)
            # so in general we want to add +1 to ensure we always keep T0
            # in case of max_history_num_frames=0 we effectively leave only T0
            # ego
            agents_polys_step[:, 0, self._history_num_frames_ego + 1:] = 0
            agents_avail_step[:, 0, self._history_num_frames_ego + 1:] = 0
            # agents
            agents_polys_step[:, 1:, self._history_num_frames_agents + 1:] = 0
            agents_avail_step[:, 1:, self._history_num_frames_agents + 1:] = 0

            # transform agents and statics into right coordinate system (ts)
            agents_polys_step = transform_points(agents_polys_step, ts_from_t0, agents_avail_step, yaw_ts_from_t0)
            static_avail_step = static_avail.clone()
            static_polys_step = transform_points(static_polys.clone(), ts_from_t0, static_avail_step)

            # get predictions and attention of the model
            out, attn = self.model_call(
                agents_polys_step,
                static_polys_step,
                agents_avail_step,
                static_avail_step,
                type_embedding,
                lane_bdry_len,
            )

            # outputs are in ts space (optionally xy normalised)
            pred_xy_step = out[:, 0, :2]
            pred_yaw_step = out[:, 0, 2:3] if not self.limit_predicted_yaw else 0.3 * torch.tanh(out[:, 0, 2:3])

            pred_xy_step_unnorm = pred_xy_step
            if self.normalize_targets:
                pred_xy_step_unnorm = pred_xy_step * self.xy_scale[0]

            # ==== SAVE PREDICTIONS & GT
            gt_xy_step_ts = data_batch["target_positions"][:, idx: idx + 1] @ ts_from_t0[..., :2, :2].transpose(
                1, 2
            ) + ts_from_t0[..., :2, -1:].transpose(1, 2)
            gt_xy_step_ts = gt_xy_step_ts[:, 0]
            gt_yaw_ts = data_batch["target_yaws"][:, idx] + yaw_ts_from_t0

            if self.normalize_targets:
                gt_xy_step_ts = gt_xy_step_ts / self.xy_scale[0]

            pred_xy_step_t0 = pred_xy_step_unnorm[:, None, :] @ t0_from_ts[..., :2, :2].transpose(1, 2) + t0_from_ts[
                ..., :2, -1:
            ].transpose(1, 2)
            pred_xy_step_t0 = pred_xy_step_t0[:, 0]
            pred_yaw_step_t0 = pred_yaw_step + yaw_t0_from_ts

            outputs_ts.append(torch.cat([pred_xy_step, pred_yaw_step], -1))
            outputs_t0.append(torch.cat([pred_xy_step_t0, pred_yaw_step_t0], -1))
            gts_ts.append(torch.cat([gt_xy_step_ts, gt_yaw_ts], -1))
            if attn is not None:
                attns.append(attn)

            # clone as we might change in place
            pred_xy_step_unnorm = pred_xy_step_unnorm.clone()
            pred_yaw_step = pred_yaw_step.clone()

            # ==== UPDATE HISTORY WITH INFORMATION FROM PREDICTION

            # update transformation matrices
            t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0 = self.update_transformation_matrices(
                pred_xy_step_unnorm, pred_yaw_step, t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0, zero, one
            )

            # update AoI
            agents_polys[:, 0, current_timestep + 1, :2] = pred_xy_step_t0
            agents_polys[:, 0, current_timestep + 1, 2:3] = pred_yaw_step_t0
            agents_avail[:, 0, current_timestep + 1] = 1

            # move time window one step into the future
            current_timestep += 1

            # detach if requested, or if in initial sampling phase
            if self.detach_unroll or idx < self.warmup_num_frames:
                t0_from_ts.detach_()
                ts_from_t0.detach_()
                yaw_t0_from_ts.detach_()
                yaw_ts_from_t0.detach_()
                agents_polys.detach_()
                static_polys.detach_()
                agents_avail.detach_()
                static_avail.detach_()

        # recombine predictions
        outputs_ts = torch.stack(outputs_ts, dim=1)
        outputs_t0 = torch.stack(outputs_t0, dim=1)
        targets = torch.stack(gts_ts, dim=1)
        attns = torch.cat(attns, dim=1)

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, future_num_frames, 3]
            target_weights = data_batch["target_availabilities"][:, :future_num_frames]
            # only calculate loss for the correct frames, i.e. not during the warmup phase,
            target_weights[:, :self.warmup_num_frames] = 0
            target_weights = target_weights.unsqueeze(-1) * self.weights_scaling

            # discount timesteps t via discount_factor**t
            # [future_num_frames]
            discount_weights = torch.tensor([self.discount_factor**(t - self.warmup_num_frames) for t in range(
                target_weights.shape[1])])
            target_weights *= discount_weights[None, ..., None].to(target_weights.device)
            loss = torch.mean(self.criterion(outputs_ts, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            # for visualisation we need results in t0
            pred_positions = outputs_t0[:, :, :2]
            pred_yaws = outputs_t0[:, :, 2:3]

            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            eval_dict["attention_weights"] = attns
            return eval_dict

    def update_transformation_matrices(self, pred_xy_step_unnorm: torch.Tensor, pred_yaw_step: torch.Tensor,
                                       t0_from_ts: torch.Tensor, ts_from_t0: torch.Tensor, yaw_t0_from_ts: torch.Tensor,
                                       yaw_ts_from_t0: torch.Tensor, zero: torch.Tensor, one: torch.Tensor
                                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Updates the used transformation matrices to reflect AoI's new position.
        """
        tr_tsplus_from_ts = -pred_xy_step_unnorm
        yaw_tsplus_from_ts = -pred_yaw_step
        yaw_ts_from_tsplus = pred_yaw_step

        # NOTE: these are full roto-translation matrices. We use the closed form and not invert for performance reasons.
        # tsplus_from_ts will bring the current predictions at ts into 0.
        tsplus_from_ts = torch.cat(
            [
                yaw_tsplus_from_ts.cos(),
                -yaw_tsplus_from_ts.sin(),
                tr_tsplus_from_ts[:, :1] * yaw_tsplus_from_ts.cos()
                - tr_tsplus_from_ts[:, 1:] * yaw_tsplus_from_ts.sin(),
                yaw_tsplus_from_ts.sin(),
                yaw_tsplus_from_ts.cos(),
                tr_tsplus_from_ts[:, :1] * yaw_tsplus_from_ts.sin()
                + tr_tsplus_from_ts[:, 1:] * yaw_tsplus_from_ts.cos(),
                zero,
                zero,
                one,
            ],
            dim=1,
        ).view(-1, 3, 3)
        # this is only required to keep t0_from_ts updated
        ts_from_tsplus = torch.cat(
            [
                yaw_ts_from_tsplus.cos(),
                -yaw_ts_from_tsplus.sin(),
                -tr_tsplus_from_ts[:, :1],
                yaw_ts_from_tsplus.sin(),
                yaw_ts_from_tsplus.cos(),
                -tr_tsplus_from_ts[:, 1:],
                zero,
                zero,
                one,
            ],
            dim=1,
        ).view(-1, 3, 3)

        # update RTs and yaws by including tsplus (next step ts)
        t0_from_ts = t0_from_ts @ ts_from_tsplus
        ts_from_t0 = tsplus_from_ts @ ts_from_t0
        yaw_t0_from_ts = yaw_t0_from_ts + yaw_ts_from_tsplus
        yaw_ts_from_t0 = yaw_ts_from_t0 + yaw_tsplus_from_ts

        return t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0
