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


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes, scores, iou_threshold, sigma, min_score, method,
                offset):
        dets = boxes.new_empty((boxes.size(0), 5), device='cpu')
        inds = ext_module.softnms(boxes.cpu(),
                                  scores.cpu(),
                                  dets.cpu(),
                                  iou_threshold=float(iou_threshold),
                                  sigma=float(sigma),
                                  min_score=float(min_score),
                                  method=int(method),
                                  offset=int(offset))
        return dets, inds


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


def softnms(boxes,
            scores,
            iou_threshold=0.3,
            sigma=0.5,
            min_score=1e-3,
            method='linear',
            offset=0):
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, other CPU NMS will be used.
    The return type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.array): boxes in shape (N, 4).
        scores (torch.Tensor or np.array): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        sigma (float): hyperparameter for gaussian method.
        min_score (float): score filter threshold.
        method (str): either 'linear' or 'gaussian'.
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
    method_dict = {'naive': 0, 'linear': 1, 'gaussian': 2}
    assert method in method_dict.keys()

    dets, inds = SoftNMSop.apply(boxes.cpu(), scores.cpu(),
                                 float(iou_threshold), float(sigma),
                                 float(min_score), method_dict[method],
                                 int(offset))
    dets = dets[:inds.size(0)]
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds
