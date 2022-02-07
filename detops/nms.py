import numpy as np
import torch

import detops._ext as ext_module


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, offset):
        inds = ext_module.nms(bboxes,
                              scores,
                              iou_threshold=float(iou_threshold),
                              offset=offset)
        return inds


def nms(boxes, scores, iou_threshold, offset=0):
    """Dispatch to either CPU or GPU implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, other CPU NMS will be used.
    The return type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.array): boxes in shape (N, 4).
        scores (torch.Tensor or np.array): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).

    Returns:
        tuple: kept dets(boxes and scores) and indices, witch is always the
            same data type as the input.
    """
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = NMSop.apply(boxes, scores, iou_threshold, offset)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=-1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds
