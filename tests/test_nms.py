import numpy as np
import torch

from detops import nms


class Testnms(object):
    def test_nms_all_close(self):
        if not torch.cuda.is_available():
            return

        np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                             [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                            dtype=np.float32)
        np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
        np_inds = np.array([1, 0, 3])
        np_dets = np.array([[3.0, 6.0, 9.0, 11.0, 0.9],
                            [6.0, 3.0, 8.0, 7.0, 0.6],
                            [1.0, 4.0, 13.0, 7.0, 0.2]])
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        dets, inds = nms(boxes, scores, iou_threshold=0.3, offset=0)
        assert np.allclose(dets, np_dets)  # test cpu
        assert np.allclose(inds, np_inds)  # test cpu
        dets, inds = nms(boxes.cuda(),
                         scores.cuda(),
                         iou_threshold=0.3,
                         offset=0)
        assert np.allclose(dets.cpu().numpy(), np_dets)  # test gpu
        assert np.allclose(inds.cpu().numpy(), np_inds)  # test gpu
