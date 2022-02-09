import torch
from torch.nn.modules.utils import _pair

import detops._ext as ext_module


class RoIAlignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):

        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('avg', 'max')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2).'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)

        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        ext_module.roi_align_forward(input,
                                     rois,
                                     output,
                                     argmax_y,
                                     argmax_x,
                                     aligned_height=ctx.output_size[0],
                                     aligned_width=ctx.output_size[1],
                                     spatial_scale=ctx.spatial_scale,
                                     sampling_ratio=ctx.sampling_ratio,
                                     pool_mode=ctx.pool_mode,
                                     aligned=ctx.aligned)

        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        grad_output = grad_output.contiguous()
        ext_module.roi_align_backward(grad_output,
                                      rois,
                                      argmax_y,
                                      argmax_x,
                                      grad_input,
                                      aligned_height=ctx.output_size[0],
                                      aligned_width=ctx.output_size[1],
                                      spatial_scale=ctx.spatial_scale,
                                      sampling_ratio=ctx.sampling_ratio,
                                      pool_mode=ctx.pool_mode,
                                      aligned=ctx.aligned)

        return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(torch.nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): (h, w).
        spatial_scale (float): scale the input boxes by this number.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode for each bin.
        aligned (bool): align or not.
    """
    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True):
        super(RoIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input (torch.Tensor): NCHW images.
            rois (torch.Tensor): B x 5 boxes, (batch_idx, x1, y1, x2, y2).
        """
        return roi_align(input, rois, self.output_size, self.spatial_scale,
                         self.sampling_ratio, self.pool_mode, self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale})'
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        return s
