import torch
from gpytorch.utils.broadcasting import _mul_broadcast_shape


def _match_batch_dims(x1, *args):
    batch_shape = x1.shape[:-2]
    # Make sure the batch shapes agree for training/test data
    output = [x1]
    for x2 in args:
        if batch_shape != x1.shape[:-2]:
            batch_shape = _mul_broadcast_shape(batch_shape, x1.shape[:-2])
            x1 = x1.expand(*batch_shape, *x1.shape[-2:])
        if batch_shape != x2.shape[:-2]:
            batch_shape = _mul_broadcast_shape(batch_shape, x2.shape[:-2])
            x1 = x1.expand(*batch_shape, *x1.shape[-2:])
            x2 = x2.expand(*batch_shape, *x2.shape[-2:])
        output.append(x2)
    return output


def _match_dtype(x_in, *args):
    in_dtype = x_in.dtype
    output = []
    for arg in args:
        out = arg.to(in_dtype)
        output.append(out)
    return output
