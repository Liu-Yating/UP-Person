r"""Functional interface"""
from typing import Callable, List, Optional, Tuple
import math
import warnings

import torch
from torch import _VF
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes

from torch._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn import _reduction as _Reduction
from torch.nn import grad  # noqa: F401
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default


Tensor = torch.Tensor


def fractional_max_pool2d_with_indices(
    input: Tensor, kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    r"""Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k`)
                     or a tuple `(kH, kW)`
        output_size: the target output size of the image of the form :math:`oH \times oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool2d_with_indices,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool2d requires specifying either " "an output_size or an output_ratio")
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _pair(output_ratio)
        output_size = [int(input.size(-2) * _output_ratio[0]), int(input.size(-1) * _output_ratio[1])]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 3 else input.size(0)
        _random_samples = torch.rand(n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)


def _fractional_max_pool2d(
    input: Tensor, kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None
) -> Tensor:
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool2d,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    return fractional_max_pool2d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool2d_with_indices,
    if_false=_fractional_max_pool2d,
    module_name=__name__,
    func_name="fractional_max_pool2d",
)


def fractional_max_pool3d_with_indices(
    input: Tensor, kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    r"""Applies 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT \times kH \times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \times k \times k`)
                     or a tuple `(kT, kH, kW)`
        output_size: the target output size of the form :math:`oT \times oH \times oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output
                     :math:`oH \times oH \times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool3d_with_indices,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool3d requires specifying either " "an output_size or an output_ratio")
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [
            int(input.size(2) * _output_ratio[0]),
            int(input.size(3) * _output_ratio[1]),
            int(input.size(4) * _output_ratio[2]),
        ]

    if _random_samples is None:
        _random_samples = torch.rand(input.size(0), input.size(1), 3, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool3d(input, kernel_size, output_size, _random_samples)


def _fractional_max_pool3d(
    input: Tensor, kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None
) -> Tensor:
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(
            fractional_max_pool3d,
            (input, _random_samples),
            input,
            kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )
    return fractional_max_pool3d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool3d_with_indices,
    if_false=_fractional_max_pool3d,
    module_name=__name__,
    func_name="fractional_max_pool3d",
)


def max_pool1d_with_indices(
    input: Tensor, kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool1d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool1d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


def _max_pool1d(
    input: Tensor, kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool1d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool1d_with_indices,
    if_false=_max_pool1d,
    module_name=__name__,
    func_name="max_pool1d",
)


def max_pool2d_with_indices(
    input: Tensor, kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool2d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch._C._nn.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


def _max_pool2d(
    input: Tensor, kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool2d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool2d_with_indices,
    if_false=_max_pool2d,
    module_name=__name__,
    func_name="max_pool2d",
)


def max_pool3d_with_indices(
    input: Tensor, kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool3d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool3d_with_indices,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch._C._nn.max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


def _max_pool3d(
    input: Tensor, kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_pool3d,
            (input,),
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=return_indices,
        )
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool3d_with_indices,
    if_false=_max_pool3d,
    module_name=__name__,
    func_name="max_pool3d",
)


def _unpool_output_size(
    input: Tensor, kernel_size: List[int], stride: List[int], padding: List[int], output_size: Optional[List[int]]
) -> List[int]:
    input_size = input.size()
    default_size = torch.jit.annotate(List[int], [])
    for d in range(len(kernel_size)):
        default_size.append((input_size[-len(kernel_size) + d] - 1) * stride[d] + kernel_size[d] - 2 * padding[d])
    if output_size is None:
        ret = default_size
    else:
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]
        if len(output_size) != len(kernel_size):
            raise ValueError(
                "output_size should be a sequence containing "
                "{} or {} elements, but it has a length of '{}'".format(
                    len(kernel_size), len(kernel_size) + 2, len(output_size)
                )
            )
        for d in range(len(kernel_size)):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not (min_size < output_size[d] < max_size):
                raise ValueError(
                    'invalid output_size "{}" (dim {} must be between {} and {})'.format(
                        output_size, d, min_size, max_size
                    )
                )

        ret = output_size
    return ret


def max_unpool1d(
    input: Tensor, indices: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    output_size: Optional[BroadcastingList1[int]] = None
) -> Tensor:
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool1d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _single(kernel_size)
    if stride is not None:
        _stride = _single(stride)
    else:
        _stride = kernel_size
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    if isinstance(output_size, list):
        output_size = output_size + [1]
    else:
        output_size = output_size + (1,)
    return torch._C._nn.max_unpool2d(input.unsqueeze(-1), indices.unsqueeze(-1), output_size).squeeze(-1)


def max_unpool2d(
    input: Tensor, indices: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    output_size: Optional[BroadcastingList2[int]] = None
) -> Tensor:
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool2d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _pair(kernel_size)
    if stride is not None:
        _stride = _pair(stride)
    else:
        _stride = kernel_size
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return torch._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(
    input: Tensor, indices: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    output_size: Optional[BroadcastingList3[int]] = None
) -> Tensor:
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool3d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _triple(kernel_size)
    if stride is not None:
        _stride = _triple(stride)
    else:
        _stride = kernel_size
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return torch._C._nn.max_unpool3d(input, indices, output_size, _stride, padding)


def lp_pool2d(
    input: Tensor, norm_type: float,
    kernel_size: int,
    stride: Optional[BroadcastingList2[int]] = None,
    ceil_mode: bool = False
) -> Tensor:
    r"""Applies a 2D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool2d, (input,), input, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode
        )
    kw, kh = utils._pair(kernel_size)
    if stride is not None:
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1.0 / norm_type)


def lp_pool1d(
    input: Tensor, norm_type: float,
    kernel_size: int,
    stride: Optional[BroadcastingList1[int]] = None,
    ceil_mode: bool = False
) -> Tensor:
    r"""Applies a 1D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool1d, (input,), input, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode
        )
    if stride is not None:
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1.0 / norm_type)


def adaptive_max_pool1d_with_indices(
    input: Tensor, output_size: BroadcastingList1[int], return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool1d_with_indices, (input,), input, output_size, return_indices=return_indices
        )
    return torch.adaptive_max_pool1d(input, output_size)


def _adaptive_max_pool1d(input: Tensor, output_size: BroadcastingList1[int], return_indices: bool = False) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool1d, (input,), input, output_size, return_indices=return_indices
        )
    return adaptive_max_pool1d_with_indices(input, output_size)[0]


adaptive_max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool1d_with_indices,
    if_false=_adaptive_max_pool1d,
    module_name=__name__,
    func_name="adaptive_max_pool1d",
)


def adaptive_max_pool2d_with_indices(
    input: Tensor, output_size: BroadcastingList2[int],
    return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool2d_with_indices, (input,), input, output_size, return_indices=return_indices
        )
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool2d(input, output_size)


def _adaptive_max_pool2d(input: Tensor, output_size: BroadcastingList2[int], return_indices: bool = False) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool2d, (input,), input, output_size, return_indices=return_indices
        )
    return adaptive_max_pool2d_with_indices(input, output_size)[0]


adaptive_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool2d_with_indices,
    if_false=_adaptive_max_pool2d,
    module_name=__name__,
    func_name="adaptive_max_pool2d",
)


def adaptive_max_pool3d_with_indices(
    input: Tensor, output_size: BroadcastingList3[int],
    return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool3d_with_indices, (input,), input, output_size, return_indices=return_indices
        )
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool3d(input, output_size)


def _adaptive_max_pool3d(input: Tensor, output_size: BroadcastingList3[int], return_indices: bool = False) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            adaptive_max_pool3d, (input,), input, output_size, return_indices=return_indices
        )
    return adaptive_max_pool3d_with_indices(input, output_size)[0]


adaptive_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool3d_with_indices,
    if_false=_adaptive_max_pool3d,
    module_name=__name__,
    func_name="adaptive_max_pool3d",
)




def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_avg_pool2d, (input,), input, output_size)
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool2d(input, _output_size)


def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList3[int]) -> Tensor:
    r"""
    Applies a 3D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
    if has_torch_function_unary(input):
        return handle_torch_function(adaptive_avg_pool3d, (input,), input, output_size)
    _output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_avg_pool3d(input, _output_size)


# Activation functions
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(dropout, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)


def alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False) -> Tensor:
    r"""Applies alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(alpha_dropout, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.alpha_dropout_(input, p, training) if inplace else _VF.alpha_dropout(input, p, training)


def dropout2d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(dropout2d, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.feature_dropout_(input, p, training) if inplace else _VF.feature_dropout(input, p, training)


def dropout3d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    # This is 100% the same code as dropout2d. We duplicate this code so that
    # stack traces are not confusing.
    if has_torch_function_unary(input):
        return handle_torch_function(dropout3d, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.feature_dropout_(input, p, training) if inplace else _VF.feature_dropout(input, p, training)


def feature_alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False) -> Tensor:
    r"""
    Randomly masks out entire channels (a channel is a feature map,
    e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]`) of the input tensor). Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function.

    Each element will be masked independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    See :class:`~torch.nn.FeatureAlphaDropout` for details.

    Args:
        p: dropout probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            feature_alpha_dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.feature_alpha_dropout_(input, p, training) if inplace else _VF.feature_alpha_dropout(input, p, training)


def _threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    r"""Thresholds each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(_threshold, (input,), input, threshold, value, inplace=inplace)
    if inplace:
        result = _VF.threshold_(input, threshold, value)
    else:
        result = _VF.threshold(input, threshold, value)
    return result


# We define this function as _threshold because it takes an argument
# named threshold, which clobbers the recursive reference to the
# function needed for __torch_function__ support
threshold = _threshold




def relu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(relu, (input,), input, inplace=inplace)
    if inplace:
        result = torch.relu_(input)
    else:
        result = torch.relu(input)
    return result




def glu(input: Tensor, dim: int = -1) -> Tensor:
    r"""
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::
        \text{GLU}(a, b) = a \otimes \sigma(b)

    where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
    is the sigmoid function and :math:`\otimes` is the element-wise product between matrices.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input. Default: -1
    """
    if has_torch_function_unary(input):
        return handle_torch_function(glu, (input,), input, dim=dim)
    if input.dim() == 0:
        raise RuntimeError("glu does not support scalars because halving size must be even")
    return torch._C._nn.glu(input, dim)


def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    r"""
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardtanh, (input,), input, min_val=min_val, max_val=max_val, inplace=inplace)
    if inplace:
        result = torch._C._nn.hardtanh_(input, min_val, max_val)
    else:
        result = torch._C._nn.hardtanh(input, min_val, max_val)
    return result




def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    r"""relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(relu6, (input,), input, inplace=inplace)
    if inplace:
        result = torch._C._nn.relu6_(input)
    else:
        result = torch._C._nn.relu6(input)
    return result


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""Applies element-wise,
    :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`.

    See :class:`~torch.nn.ELU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(elu, (input,), input, alpha=alpha, inplace=inplace)
    if inplace:
        result = torch._C._nn.elu_(input, alpha)
    else:
        result = torch._C._nn.elu(input, alpha)
    return result



def selu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))`,
    with :math:`\alpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(selu, (input,), input, inplace=inplace)
    if inplace:
        result = torch.selu_(input)
    else:
        result = torch.selu(input)
    return result




def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""celu(input, alpha=1., inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))`.

    See :class:`~torch.nn.CELU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(celu, (input,), input, alpha=alpha, inplace=inplace)
    if inplace:
        result = torch.celu_(input, alpha)
    else:
        result = torch.celu(input, alpha)
    return result





def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(leaky_relu, (input,), input, negative_slope=negative_slope, inplace=inplace)
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    else:
        result = torch._C._nn.leaky_relu(input, negative_slope)
    return result




def prelu(input: Tensor, weight: Tensor) -> Tensor:
    r"""prelu(input, weight) -> Tensor

    Applies element-wise the function
    :math:`\text{PReLU}(x) = \max(0,x) + \text{weight} * \min(0,x)` where weight is a
    learnable parameter.

    See :class:`~torch.nn.PReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(prelu, (input,), input, weight)
    return torch.prelu(input, weight)


def rrelu(
    input: Tensor, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = False, inplace: bool = False
) -> Tensor:
    r"""rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

    Randomized leaky ReLU.

    See :class:`~torch.nn.RReLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            rrelu, (input,), input, lower=lower, upper=upper, training=training, inplace=inplace
        )
    if inplace:
        result = torch.rrelu_(input, lower, upper, training)
    else:
        result = torch.rrelu(input, lower, upper, training)
    return result




def gelu(input):
    r"""gelu(input) -> Tensor

    Applies element-wise the function
    :math:`\text{GELU}(x) = x * \Phi(x)`

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(gelu, (input,), input)
    return torch._C._nn.gelu(input)


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    r"""
    hardshrink(input, lambd=0.5) -> Tensor

    Applies the hard shrinkage function element-wise

    See :class:`~torch.nn.Hardshrink` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardshrink, (input,), input, lambd=lambd)
    return torch.hardshrink(input, lambd)


def tanhshrink(input):
    r"""tanhshrink(input) -> Tensor

    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(tanhshrink, (input,), input)
    return input - input.tanh()


def softsign(input):
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(softsign, (input,), input)
    return input / (input.abs() + 1)



def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmin(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmin function.

    Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`. See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(softmin, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim("softmin", input.dim(), _stacklevel)
    if dtype is None:
        ret = (-input).softmax(dim)
    else:
        ret = (-input).softmax(dim, dtype=dtype)
    return ret


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    if has_torch_function_unary(input):
        return handle_torch_function(softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Args:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is cast to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(log_softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim("log_softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.log_softmax(dim)
    else:
        ret = input.log_softmax(dim, dtype=dtype)
    return ret




def tanh(input):
    r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
    return input.tanh()


def sigmoid(input):
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    return input.sigmoid()


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Applies the element-wise function

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    See :class:`~torch.nn.Hardsigmoid` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardsigmoid, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.hardsigmoid_(input)
    return torch._C._nn.hardsigmoid(input)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(linear, (input, weight, bias), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)


def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`

    Shape:

        - input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}`
          and :math:`*` means any number of additional dimensions.
          All but the last dimension of the inputs should be the same.
        - input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`
        - weight: :math:`(\text{out\_features}, \text{in1\_features},
          \text{in2\_features})`
        - bias: :math:`(\text{out\_features})`
        - output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.
    """
    if has_torch_function_variadic(input1, input2, weight, bias):
        return handle_torch_function(
            bilinear,
            (input1, input2, weight, bias),
            input1, input2, weight,
            bias=bias
        )
    return torch.bilinear(input1, input2, weight, bias)


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(silu, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.silu_(input)
    return torch._C._nn.silu(input)


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Applies the Mish function, element-wise.
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    See :class:`~torch.nn.Mish` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(mish, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.mish_(input)
    return torch._C._nn.mish(input)


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Applies the hardswish function, element-wise, as described in the paper:

    `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    See :class:`~torch.nn.Hardswish` for more details.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    if has_torch_function_unary(input):
        return handle_torch_function(hardswish, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.hardswish_(input)
    return torch._C._nn.hardswish(input)


def _no_grad_embedding_renorm_(weight: Tensor, input: Tensor, max_norm: float, norm_type: float) -> Tensor:
    with torch.no_grad():
        torch.embedding_renorm_(weight, input, max_norm, norm_type)


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0,2,0,5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """

    if has_torch_function_variadic(input, weight):
        return handle_torch_function(
            embedding, (input, weight),
            input, weight, padding_idx, max_norm, norm_type,
            scale_grad_by_freq, sparse
        )
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
    r"""Computes sums, means or maxes of `bags` of embeddings, without instantiating the
    intermediate embeddings.

    See :class:`torch.nn.EmbeddingBag` for more details.

    Note:
        {backward_reproducibility_note}

    Args:
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                             the starting index position of each bag (sequence) in :attr:`input`.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.
                                 Note: this option is not supported when ``mode="max"``.
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not None.

        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
            The last element is the size of the input, or the ending index position of the last bag (sequence).

        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". Note that the embedding
                                     vector at :attr:`padding_idx` is excluded from the reduction.

    Shape:
        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
            each of fixed length ``N``, and this will return ``B`` values aggregated in a way
            depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
            multiple bags (sequences). :attr:`offsets` is required to be a 1D tensor containing
            the starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets`
            of shape `(B)`, :attr:`input` will be viewed as having ``B`` bags.
            Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`

        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as :attr:`input`.

        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1,2,4,5,4,3,2,9])
        >>> offsets = torch.tensor([0,4])
        >>> F.embedding_bag(input, embedding_matrix, offsets)
        tensor([[ 0.3397,  0.3552,  0.5545],
                [ 0.5893,  0.4386,  0.5882]])

        >>> # example with padding_idx
        >>> embedding_matrix = torch.rand(10, 3)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0,4])
        >>> F.embedding_bag(input, embedding_matrix, offsets, padding_idx=2, mode='sum')
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])
    """
    if has_torch_function_variadic(input, weight, offsets, per_sample_weights):
        return handle_torch_function(
            embedding_bag,
            (input, weight, offsets, per_sample_weights),
            input,
            weight,
            offsets=offsets,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == torch.long and input.is_floating_point():
        warnings.warn(
            "Argument order of nn.functional.embedding_bag was changed. "
            "Usage `embedding_bag(weight, input, ...)` is deprecated, "
            "and should now be `embedding_bag(input, weight, ...)`."
        )
        weight, input = input, weight

    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError(
            "embedding_bag: If per_sample_weights ({}) is not None, "
            "then it must have the same shape as the input ({})".format(per_sample_weights.shape, input.shape)
        )

    if input.dim() == 2:
        if offsets is not None:
            type_str = "<unknown>"
            # TODO: Remove this once script supports type() calls
            if not torch.jit.is_scripting():
                type_str = str(type(offsets))
            raise ValueError(
                "if input is 2D, then offsets has to be None"
                ", as input is treated is a mini-batch of"
                " fixed length sequences. However, found "
                "offsets of type {}".format(type_str)
            )
        offsets = torch.arange(0, input.numel(), input.size(1), dtype=input.dtype, device=input.device)

        input = input.reshape(-1)
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
    else:
        raise ValueError("input has to be 1D or 2D Tensor," " but got Tensor of dimension {}".format(input.dim()))
    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError("max mode does not support scaling the gradient by the frequency")

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.nembedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    if per_sample_weights is not None and mode != "sum":
        raise NotImplementedError(
            "embedding_bag: per_sample_weights was not None. "
            "per_sample_weights is only supported for mode='sum' "
            "(got mode='{}'). Please open a feature request on GitHub.".format(mode)
        )

    ret, _, _, _ = torch.embedding_bag(
        weight, input, offsets, scale_grad_by_freq, mode_enum, sparse, per_sample_weights, include_last_offset, padding_idx
    )
    return ret


embedding_bag.__doc__ = embedding_bag.__doc__.format(**reproducibility_notes)


def _verify_batch_size(size: List[int]) -> None:
    # XXX: JIT script does not support the reduce from functools, and mul op is a
    # builtin, which cannot be used as a value to a func yet, so rewrite this size
    # check to a simple equivalent for loop
    #
    # TODO: make use of reduce like below when JIT is ready with the missing features:
    # from operator import mul
    # from functools import reduce
    #
    #   if reduce(mul, size[2:], size[0]) == 1
    size_prods = size[0]
    for i in range(len(size) - 2):
        size_prods *= size[i + 2]
    if size_prods == 1:
        raise ValueError("Expected more than 1 value per channel when training, got input size {}".format(size))


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    if has_torch_function_variadic(input, running_mean, running_var, weight, bias):
        return handle_torch_function(
            batch_norm,
            (input, running_mean, running_var, weight, bias),
            input,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )
    if training:
        _verify_batch_size(input.size())

    return torch.batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled
    )


def _verify_spatial_size(size: List[int]) -> None:
    # Verify that there is > 1 spatial element for instance norm calculation.
    size_prods = 1
    for i in range(2, len(size)):
        size_prods *= size[i]
    if size_prods == 1:
        raise ValueError("Expected more than 1 spatial element when training, got input size {}".format(size))


def instance_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Instance Normalization for each channel in each data sample in a
    batch.

    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
    :class:`~torch.nn.InstanceNorm3d` for details.
    """
    if has_torch_function_variadic(input, running_mean, running_var, weight, bias):
        return handle_torch_function(
            instance_norm,
            (input, running_mean, running_var, weight, bias),
            input,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            use_input_stats=use_input_stats,
            momentum=momentum,
            eps=eps,
        )
    if use_input_stats:
        _verify_spatial_size(input.size())
    return torch.instance_norm(
        input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, torch.backends.cudnn.enabled
    )


def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            layer_norm, (input, weight, bias), input, normalized_shape, weight=weight, bias=bias, eps=eps
        )
    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)


def group_norm(
    input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-5
) -> Tensor:
    r"""Applies Group Normalization for last certain number of dimensions.

    See :class:`~torch.nn.GroupNorm` for details.
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(group_norm, (input, weight, bias,), input, num_groups, weight=weight, bias=bias, eps=eps)
    _verify_batch_size([input.size(0) * input.size(1) // num_groups, num_groups] + list(input.size()[2:]))
    return torch.group_norm(input, num_groups, weight, bias, eps, torch.backends.cudnn.enabled)


def local_response_norm(input: Tensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> Tensor:
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(local_response_norm, (input,), input, size, alpha=alpha, beta=beta, k=k)
    dim = input.dim()
    if dim < 3:
        raise ValueError(
            "Expected 3D or higher dimensionality \
                         input (got {} dimensions)".format(
                dim
            )
        )

    if input.numel() == 0:
        return input

    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div


# loss


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    r"""The Connectionist Temporal Classification loss.

    See :class:`~torch.nn.CTCLoss` for details.

    Note:
        {cudnn_reproducibility_note}

    Note:
        {backward_reproducibility_note}

    Args:
        log_probs: :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`.
            Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
        blank (int, optional):
            Blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Example::

        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()
    """
    if has_torch_function_variadic(log_probs, targets, input_lengths, target_lengths):
        return handle_torch_function(
            ctc_loss,
            (log_probs, targets, input_lengths, target_lengths),
            log_probs, targets, input_lengths, target_lengths,
            blank=blank, reduction=reduction, zero_infinity=zero_infinity
        )
    return torch.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank, _Reduction.get_enum(reduction), zero_infinity
    )


ctc_loss.__doc__ = ctc_loss.__doc__.format(**reproducibility_notes)


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to be log-probabilities.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            nll_loss,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return torch._C._nn.nll_loss_nd(input, target, weight, _Reduction.get_enum(reduction), ignore_index)


def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    eps: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: expectation of underlying Poisson distribution.
        target: random sample :math:`target \sim \text{Poisson}(input)`.
        log_input: if ``True`` the loss is computed as
            :math:`\exp(\text{input}) - \text{target} * \text{input}`, if ``False`` then loss is
            :math:`\text{input} - \text{target} * \log(\text{input}+\text{eps})`. Default: ``True``
        full: whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            :math:`\text{target} * \log(\text{target}) - \text{target} + 0.5 * \log(2 * \pi * \text{target})`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input`\ =\ ``False``. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            poisson_nll_loss,
            (input, target),
            input,
            target,
            log_input=log_input,
            full=full,
            size_average=size_average,
            eps=eps,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        ret = input
        raise ValueError(reduction + " is not valid")

    ret = torch.poisson_nll_loss(input, target, log_input, full, eps, _Reduction.get_enum(reduction))
    return ret


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target, var):
        return handle_torch_function(
            gaussian_nll_loss,
            (input, target, var),
            input,
            target,
            var,
            full=full,
            eps=eps,
            reduction=reduction,
        )

    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    r"""The `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Tensor of arbitrary shape in log-probabilities.
        target: Tensor of the same shape as input. See :attr:`log_target` for
            the target's interpretation.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'mean'``
        log_target (bool): A flag indicating whether ``target`` is passed in the log space.
            It is recommended to pass certain distributions (like ``softmax``)
            in the log space to avoid numerical issues caused by explicit ``log``.
            Default: ``False``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. note::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
        In the next major release, ``'mean'`` will be changed to be the same as 'batchmean'.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            kl_div,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            log_target=log_target,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        if reduction == "mean":
            warnings.warn(
                "reduction: 'mean' divides the total loss by both the batch size and the support size."
                "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
                "'mean' will be changed to behave the same as 'batchmean' in the next major release."
            )

        # special case for batchmean
        if reduction == "batchmean":
            reduction_enum = _Reduction.get_enum("sum")
        else:
            reduction_enum = _Reduction.get_enum(reduction)

    reduced = torch.kl_div(input, target, reduction_enum, log_target=log_target)

    if reduction == "batchmean" and input.dim() != 0:
        reduced = reduced / input.size()[0]

    return reduced


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""This criterion computes the cross entropy loss between input and target.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
            Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Examples::

        >>> # Example of target with class indices
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Function that measures the Binary Cross Entropy between the target and input
    probabilities.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape as probabilities.
        target: Tensor of the same shape as input with values between 0 and 1.
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            binary_cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if target.size() != input.size():
        raise ValueError(
            "Using a target size ({}) that is different to the input size ({}) is deprecated. "
            "Please ensure they have the same size.".format(target.size(), input.size())
        )

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    return torch._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Function that measures Binary Cross Entropy between target and input
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        target: Tensor of the same shape as input with values between 0 and 1
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if has_torch_function_variadic(input, target, weight, pos_weight):
        return handle_torch_function(
            binary_cross_entropy_with_logits,
            (input, target, weight, pos_weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    return torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    r"""Function that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            smooth_l1_loss,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            beta=beta,
        )
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.smooth_l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction), beta)


def huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    delta: float = 1.0,
) -> Tensor:
    r"""Function that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.

    See :class:`~torch.nn.HuberLoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            huber_loss,
            (input, target),
            input,
            target,
            reduction=reduction,
            delta=delta,
        )
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.huber_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction), delta)


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            l1_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))


def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            mse_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))


def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """
    if has_torch_function_variadic(input1, input2, target):
        return handle_torch_function(
            margin_ranking_loss,
            (input1, input2, target),
            input1,
            input2,
            target,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if input1.dim() == 0 or input2.dim() == 0 or target.dim() == 0:
        raise RuntimeError(
            (
                "margin_ranking_loss does not support scalars, got sizes: "
                "input1: {}, input2: {}, target: {} ".format(input1.size(), input2.size(), target.size())
            )
        )
    return torch.margin_ranking_loss(input1, input2, target, margin, reduction_enum)


def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            hinge_embedding_loss,
            (input, target),
            input,
            target,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch.hinge_embedding_loss(input, target, margin, reduction_enum)


def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            multilabel_margin_loss,
            (input, target),
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch._C._nn.multilabel_margin_loss(input, target, reduction_enum)


def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.SoftMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            soft_margin_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch._C._nn.soft_margin_loss(input, target, reduction_enum)


def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            multilabel_soft_margin_loss,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values

    if reduction == "none":
        ret = loss
    elif reduction == "mean":
        ret = loss.mean()
    elif reduction == "sum":
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.CosineEmbeddingLoss` for details.
    """
    if has_torch_function_variadic(input1, input2, target):
        return handle_torch_function(
            cosine_embedding_loss,
            (input1, input2, target),
            input1,
            input2,
            target,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch.cosine_embedding_loss(input1, input2, target, margin, reduction_enum)


def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
                          reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            multi_margin_loss,
            (input, target, weight),
            input,
            target,
            p=p,
            margin=margin,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if p != 1 and p != 2:
        raise ValueError("only p == 1 and p == 2 supported")
    if weight is not None:
        if weight.dim() != 1:
            raise ValueError("weight must be one-dimensional")

    return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction_enum)




@_overload  # noqa: F811
def upsample(input: Tensor, size: Optional[int] = None, scale_factor: Optional[float] = None, mode: str = "nearest", align_corners: Optional[bool] = None) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample(input: Tensor, size: Optional[List[int]] = None, scale_factor: Optional[float] = None, mode: str = "nearest", align_corners: Optional[bool] = None) -> Tensor:  # noqa: F811
    pass


def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):  # noqa: F811
    r"""Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(...)``.

    Note:
        {backward_reproducibility_note}

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only)

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (string): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    """
    warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode, align_corners)


upsample.__doc__ = upsample.__doc__.format(**reproducibility_notes)


@_overload  # noqa: F811
def interpolate(input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def interpolate(input: Tensor, size: Optional[List[int]] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def interpolate(input: Tensor, size: Optional[int] = None, scale_factor: Optional[float] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def interpolate(  # noqa: F811
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
) -> Tensor:  # noqa: F811
    pass

def interpolate(input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> Tensor:  # noqa: F811
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. If `scale_factor` is a tuple,
            its length has to match `input.dim()`.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recomputed_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    .. warning::
        When scale_factor is specified, if recompute_scale_factor=True,
        scale_factor is used to compute the output_size which will then
        be used to infer new scales for the interpolation.
        The default behavior for recompute_scale_factor changed to False
        in 1.6.0, and scale_factor is used in the interpolation
        calculation.

    Note:
        {backward_reproducibility_note}
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            interpolate,
            (input,),
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
        )

    if mode in ("nearest", "area"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            warnings.warn(
                "Default upsampling behavior when mode={} is changed "
                "to align_corners=False since 0.4.0. Please specify "
                "align_corners=True if the old behavior is desired. "
                "See the documentation of nn.Upsample for details.".format(mode)
            )
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "size shape must match input shape. " "Input is {}D, size is {}".format(dim, len(size))
                )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "scale_factor shape must match input shape. "
                    "Input is {}D, scale_factor is {}".format(dim, len(scale_factor))
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is None:
        # only warn when the scales have floating values since
        # the result for ints is the same with/without recompute_scale_factor
        if scale_factors is not None:
            for scale in scale_factors:
                if math.floor(scale) != scale:
                    warnings.warn(
                        "The default behavior for interpolate/upsample with float scale_factor changed "
                        "in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, "
                        "instead of relying on the computed output size. "
                        "If you wish to restore the old behavior, please set recompute_scale_factor=True. "
                        "See the documentation of nn.Upsample for details. "
                    )
                    break
    elif recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()))
                for i in range(dim)
            ]
        else:
            assert scale_factors is not None
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]
        scale_factors = None

    if input.dim() == 3 and mode == "nearest":
        return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return torch._C._nn.upsample_linear1d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        return torch._C._nn.upsample_bilinear2d(input, output_size, align_corners, scale_factors)
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return torch._C._nn.upsample_trilinear3d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        return torch._C._nn.upsample_bicubic2d(input, output_size, align_corners, scale_factors)

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        " (got {}D) for the modes: nearest | linear | bilinear | bicubic | trilinear"
        " (got {})".format(input.dim(), mode)
    )


interpolate.__doc__ = interpolate.__doc__.format(**reproducibility_notes)


@_overload  # noqa: F811
def upsample_nearest(input: Tensor, size: Optional[int] = None, scale_factor: Optional[float] = None) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_nearest(input: Tensor, size: Optional[List[int]] = None, scale_factor: Optional[float] = None) -> Tensor:  # noqa: F811
    pass


def upsample_nearest(input, size=None, scale_factor=None):  # noqa: F811
    r"""Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.

    Currently spatial and volumetric upsampling are supported (i.e. expected
    inputs are 4 or 5 dimensional).

    Args:
        input (Tensor): input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatia
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.

    Note:
        {backward_reproducibility_note}
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode="nearest")


upsample_nearest.__doc__ = upsample_nearest.__doc__.format(**reproducibility_notes)


@_overload  # noqa: F811
def upsample_bilinear(
    input: Tensor, size: Optional[int] = None, scale_factor: Optional[float] = None
) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_bilinear(  # noqa: F811
    input: Tensor, size: Optional[List[int]] = None, scale_factor: Optional[float] = None
) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_bilinear(  # noqa: F811
    input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None
) -> Tensor:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_bilinear(  # noqa: F811
    input: Tensor, size: Optional[List[int]] = None, scale_factor: Optional[List[float]] = None
) -> Tensor:  # noqa: F811
    pass


def upsample_bilinear(input, size=None, scale_factor=None):  # noqa: F811
    r"""Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with
        ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` fo
    volumetric (5 dimensional) inputs.

    Args:
        input (Tensor): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size

    Note:
        {backward_reproducibility_note}
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


upsample_bilinear.__doc__ = upsample_bilinear.__doc__.format(**reproducibility_notes)

GRID_SAMPLE_INTERPOLATION_MODES = {
    "bilinear": 0,
    "nearest": 1,
    "bicubic": 2,
}

GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,
    "border": 1,
    "reflection": 2,
}


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    r"""Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_\text{in}, W_\text{in})` and :attr:`grid` with shape
    :math:`(N, H_\text{out}, W_\text{out}, 2)`, the output will have shape
    :math:`(N, C, H_\text{out}, W_\text{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
          and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes
          ``x'' = -0.5``.

    Note:
        This function is often used in conjunction with :func:`affine_grid`
        to build `Spatial Transformer Networks`_ .

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Note:
        NaN values in :attr:`grid` would be interpreted as ``-1``.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
            Note: ``mode='bicubic'`` supports only 4-D input.
            When ``mode='bilinear'`` and the input is 5-D, the interpolation mode
            used internally will actually be trilinear. However, when the input is 4-D,
            the interpolation mode will legitimately be bilinear.
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.

    .. note::
        ``mode='bicubic'`` is implemented using the `cubic convolution algorithm`_ with :math:`\alpha=-0.75`.
        The constant :math:`\alpha` might be different from packages to packages.
        For example, `PIL`_ and `OpenCV`_ use -0.5 and -0.75 respectively.
        This algorithm may "overshoot" the range of values it's interpolating.
        For example, it may produce negative values or values greater than 255 when interpolating input in [0, 255].
        Clamp the results with :func: `torch.clamp` to ensure they are within the valid range.
    .. _`cubic convolution algorithm`: https://en.wikipedia.org/wiki/Bicubic_interpolation
    .. _`PIL`: https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/src/libImaging/Resample.c#L51
    .. _`OpenCV`: https://github.com/opencv/opencv/blob/f345ed564a06178670750bad59526cfa4033be55/modules/imgproc/src/resize.cpp#L908
    """
    if has_torch_function_variadic(input, grid):
        return handle_torch_function(
            grid_sample, (input, grid), input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
    if mode != "bilinear" and mode != "nearest" and mode != "bicubic":
        raise ValueError(
            "nn.functional.grid_sample(): expected mode to be "
            "'bilinear', 'nearest' or 'bicubic', but got: '{}'".format(mode)
        )
    if padding_mode != "zeros" and padding_mode != "border" and padding_mode != "reflection":
        raise ValueError(
            "nn.functional.grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )

    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)


def affine_grid(theta: Tensor, size: List[int], align_corners: Optional[bool] = None) -> Tensor:
    r"""Generates a 2D or 3D flow field (sampling grid), given a batch of
    affine matrices :attr:`theta`.

    .. note::
        This function is often used in conjunction with :func:`grid_sample`
        to build `Spatial Transformer Networks`_ .

    Args:
        theta (Tensor): input batch of affine matrices with shape
            (:math:`N \times 2 \times 3`) for 2D or
            (:math:`N \times 3 \times 4`) for 3D
        size (torch.Size): the target output image size.
            (:math:`N \times C \times H \times W` for 2D or
            :math:`N \times C \times D \times H \times W` for 3D)
            Example: torch.Size((32, 3, 24, 24))
        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``
            to refer to the centers of the corner pixels rather than the image corners.
            Refer to :func:`grid_sample` for a more complete description.
            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`
            with the same setting for this option.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    .. warning::
        When ``align_corners = True``, 2D affine transforms on 1D data and
        3D affine transforms on 2D data (that is, when one of the spatial
        dimensions has unit size) are ill-defined, and not an intended use case.
        This is not a problem when ``align_corners = False``.
        Up to version 1.2.0, all grid points along a unit dimension were
        considered arbitrarily to be at ``-1``.
        From version 1.3.0, under ``align_corners = True`` all grid points
        along a unit dimension are considered to be at ``0``
        (the center of the input image).
    """
    if has_torch_function_unary(theta):
        return handle_torch_function(affine_grid, (theta,), theta, size, align_corners=align_corners)
    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    # enforce floating point dtype on theta
    if not theta.is_floating_point():
        raise ValueError("Expected theta to have floating point type, but got {}".format(theta.dtype))
    # check that shapes and sizes match
    if len(size) == 4:
        if theta.dim() != 3 or theta.shape[-2] != 2 or theta.shape[-1] != 3:
            raise ValueError(
                "Expected a batch of 2D affine matrices of shape Nx2x3 "
                "for size {}. Got {}.".format(size, theta.shape)
            )
        spatial_size = size[-2:]  # spatial dimension sizes
    elif len(size) == 5:
        if theta.dim() != 3 or theta.shape[-2] != 3 or theta.shape[-1] != 4:
            raise ValueError(
                "Expected a batch of 3D affine matrices of shape Nx3x4 "
                "for size {}. Got {}.".format(size, theta.shape)
            )
        spatial_size = size[-3:]  # spatial dimension sizes
    else:
        raise NotImplementedError(
            "affine_grid only supports 4D and 5D sizes, "
            "for 2D and 3D affine transforms, respectively. "
            "Got size {}.".format(size)
        )
    # check for empty span
    if align_corners and min(spatial_size) == 1:
        warnings.warn(
            "Since version 1.3.0, affine_grid behavior has changed "
            "for unit-size grids when align_corners=True. "
            "This is not an intended use case of affine_grid. "
            "See the documentation of affine_grid for details."
        )
    elif min(size) <= 0:
        raise ValueError("Expected non-zero, positive output size. Got {}".format(size))

    return torch.affine_grid_generator(theta, size, align_corners)


def _pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0.0) -> Tensor:
    r"""Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate and reflection padding is implemented for padding the last 3
        dimensions of 5D input tensor, or the last 2 dimensions of 4D input
        tensor, or the last dimension of 3D input tensor.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])

    """
    if has_torch_function_unary(input):
        return handle_torch_function(_pad, (input,), input, pad, mode=mode, value=value)
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= input.dim(), "Padding length too large"
    if mode == "constant":
        return _VF.constant_pad_nd(input, pad, value)
    else:
        assert value == 0.0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
        if len(pad) == 2 and (input.dim() == 2 or input.dim() == 3):
            if mode == "reflect":
                return torch._C._nn.reflection_pad1d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad1d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif len(pad) == 4 and (input.dim() == 3 or input.dim() == 4):
            if mode == "reflect":
                return torch._C._nn.reflection_pad2d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad2d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif len(pad) == 6 and (input.dim() == 4 or input.dim() == 5):
            if mode == "reflect":
                return torch._C._nn.reflection_pad3d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad3d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now")


# We define this function as _pad because it takes an argument
# named pad, which clobbers the recursive reference to the pad
# function needed for __torch_function__ support
pad = _pad

# distance


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> Tensor:
    r"""
    See :class:`torch.nn.PairwiseDistance` for details
    """
    if has_torch_function_variadic(x1, x2):
        return handle_torch_function(pairwise_distance, (x1, x2), x1, x2, p=p, eps=eps, keepdim=keepdim)
    return torch.pairwise_distance(x1, x2, p, eps, keepdim)




def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginLoss` for details
    """
    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_loss,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            margin=margin,
            p=p,
            eps=eps,
            swap=swap,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch.triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction_enum)


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean"
) -> Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_with_distance_loss,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            distance_function=distance_function,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )

    distance_function = distance_function if distance_function is not None else pairwise_distance

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 1:
        return output.mean()
    elif reduction_enum == 2:
        return output.sum()
    else:
        return output


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[Tensor] = None) -> Tensor:
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
    if has_torch_function_variadic(input, out):
        return handle_torch_function(normalize, (input, out), input, p=p, dim=dim, eps=eps, out=out)
    if out is None:
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, keepdim=True).clamp_min_(eps).expand_as(input)
        return torch.div(input, denom, out=out)


def assert_int_or_pair(arg: List[int], arg_name: str, message: str) -> None:
    assert isinstance(arg, int) or len(arg) == 2, message.format(arg_name)


def unfold(
    input: Tensor, kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = 1,
    padding: BroadcastingList2[int] = 0,
    stride: BroadcastingList2[int] = 1
) -> Tensor:
    r"""Extracts sliding local blocks from a batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            unfold, (input,), input, kernel_size, dilation=dilation, padding=padding, stride=stride
        )
    if input.dim() == 4:
        msg = "{} must be int or 2-tuple for 4D input"
        assert_int_or_pair(kernel_size, "kernel_size", msg)
        assert_int_or_pair(dilation, "dilation", msg)
        assert_int_or_pair(padding, "padding", msg)
        assert_int_or_pair(stride, "stride", msg)

        return torch._C._nn.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))


def fold(
    input: Tensor, output_size: BroadcastingList2[int],
    kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = 1,
    padding: BroadcastingList2[int] = 0,
    stride: BroadcastingList2[int] = 1
) -> Tensor:
    r"""Combines an array of sliding local blocks into a large containing
    tensor.

    .. warning::
        Currently, only 3-D output tensors (unfolded batched image-like tensors) are
        supported.

    See :class:`torch.nn.Fold` for details
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            fold, (input,), input, output_size, kernel_size, dilation=dilation, padding=padding, stride=stride
        )
    if input.dim() == 3:
        msg = "{} must be int or 2-tuple for 3D input"
        assert_int_or_pair(output_size, "output_size", msg)
        assert_int_or_pair(kernel_size, "kernel_size", msg)
        assert_int_or_pair(dilation, "dilation", msg)
        assert_int_or_pair(padding, "padding", msg)
        assert_int_or_pair(stride, "stride", msg)

        return torch._C._nn.col2im(
            input, _pair(output_size), _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
        )
    else:
        raise NotImplementedError("Input Error: Only 3D input Tensors are supported (got {}D)".format(input.dim()))


def _pad_circular(input: Tensor, padding: List[int]) -> Tensor:
    """Circularly pads tensor.

    Tensor values at the beginning are used to pad the end, and values at the
    end are used to pad the beginning. For example, consider a single dimension
    with values [0, 1, 2, 3]. With circular padding of (1, 1) it would be
    padded to [3, 0, 1, 2, 3, 0], and with padding (1, 2) it would be padded to
    [3, 0, 1, 2, 3, 0, 1]. If negative padding is applied then the ends of the
    tensor get removed. With circular padding of (-1, -1) the previous example
    would become [1, 2]. Circular padding of (-1, 1) would produce
    [1, 2, 3, 1].

    The first and second dimensions of the tensor are not padded.

    Args:
        input: Tensor with shape :math:`(N, C, D[, H, W])`.
        padding: Tuple containing the number of elements to pad each side of
            the tensor. The length of padding must be twice the number of
            paddable dimensions. For example, the length of padding should be 4
            for a tensor of shape :math:`(N, C, H, W)`, and the length should
            be 6 for a tensor of shape :math:`(N, C, D, H, W)`.

    Examples::

        >>> x = torch.tensor([[[[0, 1, 2], [3, 4, 5]]]])  # Create tensor
        >>> # Example 1
        >>> padding = (1, 1, 1, 1)
        >>> y = F.pad(x, padding, mode='circular')
        >>> print(y)
        tensor([[[[5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0]]]])
        >>> print(y.shape)
        torch.Size([1, 1, 4, 5])
        >>> # Example 2
        >>> padding = (1, 1, 2, 2)
        >>> z = F.pad(x, padding, mode='circular')
        >>> print(z)
        tensor([[[[2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3]]]])
        >>> print(z.shape)
        torch.Size([1, 1, 6, 5])
    """
    in_shape = input.shape
    paddable_shape = in_shape[2:]
    ndim = len(paddable_shape)

    for idx, size in enumerate(paddable_shape):
        # Only supports wrapping around once
        assert padding[-(idx * 2 + 1)] <= size, "Padding value causes wrapping around more than once."
        assert padding[-(idx * 2 + 2)] <= size, "Padding value causes wrapping around more than once."
        # Negative padding should not result in negative sizes
        assert (
            padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)] + size >= 0
        ), "Negative padding value is resulting in an empty dimension."

    # Get shape of padded tensor
    out_shape = in_shape[:2]
    for idx, size in enumerate(paddable_shape):
        out_shape += (size + padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)],)

    out = torch.empty(out_shape, dtype=input.dtype, layout=input.layout, device=input.device)

    # Put original array in padded array
    if ndim == 1:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        out[..., out_d0:out_d1] = input[..., in_d0:in_d1]
    elif ndim == 2:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        out[..., out_d0:out_d1, out_h0:out_h1] = input[..., in_d0:in_d1, in_h0:in_h1]
    elif ndim == 3:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        out_w0 = max(padding[-6], 0)
        out_w1 = out_shape[4] - max(padding[-5], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        in_w0 = max(-padding[-6], 0)
        in_w1 = in_shape[4] - max(-padding[-5], 0)

        out[..., out_d0:out_d1, out_h0:out_h1, out_w0:out_w1] = input[..., in_d0:in_d1, in_h0:in_h1, in_w0:in_w1]

    # The following steps first pad the beginning of the tensor (left side),
    # and then pad the end of the tensor (right side).
    # Note: Corners will be written more than once when ndim > 1.

    # Only in cases where padding values are > 0 are when additional copying
    # is required.

    # Pad first dimension (depth)
    if padding[-2] > 0:
        i0 = out_shape[2] - padding[-2] - max(padding[-1], 0)
        i1 = out_shape[2] - max(padding[-1], 0)
        o0 = 0
        o1 = padding[-2]
        out[:, :, o0:o1] = out[:, :, i0:i1]
    if padding[-1] > 0:
        i0 = max(padding[-2], 0)
        i1 = max(padding[-2], 0) + padding[-1]
        o0 = out_shape[2] - padding[-1]
        o1 = out_shape[2]
        out[:, :, o0:o1] = out[:, :, i0:i1]

    # Pad second dimension (height)
    if len(padding) > 2:
        if padding[-4] > 0:
            i0 = out_shape[3] - padding[-4] - max(padding[-3], 0)
            i1 = out_shape[3] - max(padding[-3], 0)
            o0 = 0
            o1 = padding[-4]
            out[:, :, :, o0:o1] = out[:, :, :, i0:i1]
        if padding[-3] > 0:
            i0 = max(padding[-4], 0)
            i1 = max(padding[-4], 0) + padding[-3]
            o0 = out_shape[3] - padding[-3]
            o1 = out_shape[3]
            out[:, :, :, o0:o1] = out[:, :, :, i0:i1]

    # Pad third dimension (width)
    if len(padding) > 4:
        if padding[-6] > 0:
            i0 = out_shape[4] - padding[-6] - max(padding[-5], 0)
            i1 = out_shape[4] - max(padding[-5], 0)
            o0 = 0
            o1 = padding[-6]
            out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]
        if padding[-5] > 0:
            i0 = max(padding[-6], 0)
            i1 = max(padding[-6], 0) + padding[-5]
            o0 = out_shape[4] - padding[-5]
            o1 = out_shape[4]
            out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]

    return out

#
# multihead attention
#

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    num_heads=None,
    prefix: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    # ori version
    # B, Nt, E = q.shape
    # q = q / math.sqrt(E)
    # # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # attn = torch.bmm(q, k.transpose(-2, -1))
    # if attn_mask is not None:
    #     attn += attn_mask
    # attn = softmax(attn, dim=-1)
    # if dropout_p > 0.0:
    #     attn = dropout(attn, p=dropout_p)
    # # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    # output = torch.bmm(attn, v)
    # return output, attn


    # prefix by lyt
    
    B, Nt, E = q.shape
    Nkv = k.shape[1]
    scale = (E // num_heads) ** -0.5
    q = q.reshape(B, Nt, num_heads, E // num_heads)
    q = q.permute(0, 2, 1, 3)
    
    k = k.reshape(B, Nkv, num_heads,  E // num_heads)
    k = k.permute(0, 2, 1, 3)
    
    v = v.reshape(B, Nkv, num_heads,  E // num_heads)
    v = v.permute(0, 2, 1, 3)
    
    attn = (q @ k.transpose(-2, -1)) * scale
    
    
    if attn_mask is not None:
        a, b, c, d = attn.shape
        attn_mask = attn_mask.reshape(a, b, c, d)
        attn += attn_mask
        
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    
    if prefix != None: 
        attn = prefix.compensate(attn) 
    
    output = (attn @ v).transpose(1, 2).reshape(B, Nt, E)
    
    
    return output, attn


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    prefix: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        #add by lyt
        if weight is not None:
            zeros = torch.zeros_like(in_proj_weight)[: -weight.shape[0]]
            weight = torch.cat([zeros, weight], dim=0).half()
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight+weight, in_proj_bias)
        else:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                #raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
                pass
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    #new add by lyt
    q = q.view(bsz, num_heads, tgt_len, head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, num_heads*head_dim)
    k = k.view(bsz, num_heads, tgt_len, head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, num_heads*head_dim)
    v = v.view(bsz, num_heads, tgt_len, head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, num_heads*head_dim)
    if prefix != None:
        k, v = prefix(k, v)
    
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, num_heads, prefix)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None



    #lora-version
    # attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    # if need_weights:
    #     # average attention weights over heads
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     return attn_output, attn_output_weights.sum(dim=1) / num_heads
    # else:
    #     return attn_output, None
