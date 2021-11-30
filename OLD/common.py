from typing import Optional

import numpy as np
import torch


def pad_points(polylines: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Pad vectors to `pad_to` size. Dimensions are:
    B: batch
    N: number of elements (polylines)
    P: number of points
    F: number of features

    :param polylines: polylines to be padded, should be (B,N,P,F) and we're padding P
    :type polylines: torch.Tensor
    :param pad_to: nums of points we want
    :type pad_to: int
    :return: the padded polylines (B,N,pad_to,F)
    :rtype: torch.Tensor
    """
    batch, num_els, num_points, num_feats = polylines.shape
    pad_len = pad_to - num_points
    pad = torch.zeros(batch, num_els, pad_len, num_feats, dtype=polylines.dtype, device=polylines.device)
    return torch.cat([polylines, pad], dim=-2)


def pad_avail(avails: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Pad avails to `pad_to` size

    :param avails: avails to be padded, should be (B,N,P) and we're padding P
    :type avails: torch.Tensor
    :param pad_to: nums of points we want
    :type pad_to: int
    :return: the padded avails (B,N,pad_to)
    :rtype: torch.Tensor
    """
    batch, num_els, num_points = avails.shape
    pad_len = pad_to - num_points
    pad = torch.zeros(batch, num_els, pad_len, dtype=avails.dtype, device=avails.device)
    return torch.cat([avails, pad], dim=-1)


def transform_points(
    element: torch.Tensor, matrix: torch.Tensor, avail: torch.Tensor, yaw: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Transform points element using the translation tr. Reapply avail afterwards to
    ensure we don't generate any "ghosts" in the past

    Args:
        element (torch.Tensor): tensor with points to transform (B,N,P,3)
        matrix (torch.Tensor): Bx3x3 RT matrices
        avail (torch.Tensor): the availability of element
        yaw (Optional[torch.Tensor]): optional yaws of the rotation matrices to apply to yaws in element

    Returns:
        torch.Tensor: the transformed tensor
    """
    tr = matrix[:, :-1, -1:].view(-1, 1, 1, 2)
    rot = matrix[:, None, :2, :2].transpose(2, 3)  # NOTE: required because we post-multiply

    # NOTE: before we did this differently - why?
    transformed_xy = element[..., :2] @ rot + tr
    transformed_yaw = element[..., 2:3]
    if yaw is not None:
        transformed_yaw = element[..., 2:3] + yaw.view(-1, 1, 1, 1)

    element = torch.cat([transformed_xy, transformed_yaw], dim=3)
    element = element * avail[..., None].clone()  # NOTE: no idea why clone is required actually
    return element


def build_target_normalization(nsteps: int) -> torch.Tensor:
    """Normalization coefficients approximated with 3-rd degree polynomials
    to avoid storing them explicitly, and allow changing the length

    :param nsteps: number of steps to generate normalisation for
    :type nsteps: int
    :return: XY scaling for the steps
    :rtype: torch.Tensor
    """

    normalization_polynomials = np.asarray(
        [
            # x scaling
            [3.28e-05, -0.0017684, 1.8088969, 2.211737],
            # y scaling
            [-5.67e-05, 0.0052056, 0.0138343, 0.0588579],  # manually decreased by 5
        ]
    )
    # assuming we predict x, y and yaw
    coefs = np.stack([np.poly1d(p)(np.arange(nsteps)) for p in normalization_polynomials])
    coefs = coefs.astype(np.float32)
    return torch.from_numpy(coefs).T
