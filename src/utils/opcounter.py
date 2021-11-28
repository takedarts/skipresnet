import collections
import functools
import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._C
import torch.jit
import torch.nn as nn
import torch.onnx

'''
Operation costs are defined as following in this implementation:

Multiply and Division operations are 1 flops each.
Exponential operation is 1 flops.
Sigmoid operation contains minus, exp, add and div (4 flops for each).
'''

LOGGER = logging.getLogger(__name__)


def _get_tensor_size(value: Any) -> Optional[List[int]]:
    if isinstance(value.type(), torch._C.TensorType):
        return value.type().sizes()
    else:
        return None


def count_operations(
    model: nn.Module,
    inputs: Tuple[torch.Tensor],
    estimators: Dict[str, Callable[
        [str, Dict[str, Any],
         List[Optional[List[int]]],
         List[Optional[List[int]]]],
        Tuple[int, List[Optional[List[int]]]]]] = {},
    strict: bool = True,
) -> int:
    # make graph
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    model.eval()
    trace = torch.jit._get_trace_graph(model, args=tuple(inputs))[0]
    graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

    # count the number of operations
    variables: Dict[int, Optional[List[int]]] = {}
    for value in graph.inputs():
        variables[value.unique()] = _get_tensor_size(value)

    default_estimators = _get_default_estimators(strict)
    default_estimators.update(estimators)
    estimators = default_estimators

    operations = sum(
        _count_operations(node, variables, estimators)
        for node in graph.nodes())

    return operations


def _count_operations(
    node: Any,  # torch._C.Node
    variables: Dict[int, Optional[List[int]]],
    estimators: Dict[str, Callable[
        [str, Dict[str, Any],
         List[Optional[List[int]]],
         List[Optional[List[int]]]],
        Tuple[int, List[Optional[List[int]]]]]],


) -> int:
    name = node.kind()
    attrs = {key: node[key] for key in node.attributeNames()}
    inputs = [variables[val.unique()] for val in node.inputs()]
    outputs = [_get_tensor_size(val) for val in node.outputs()]

    # estimate the number of operations and the shape of the output tensor
    operations, outs = estimators[name](name, attrs, inputs, outputs)

    for i in range(len(outputs)):
        if outputs[i] is not None:
            continue
        if i < len(outs) and outs[i] is not None:
            outputs[i] = outs[i]

    if LOGGER.isEnabledFor(logging.DEBUG):
        ops_text = f'{operations:,d}'
        attr_text = ', '.join(f'{k}={v}' for k, v in attrs.items())
        LOGGER.debug(
            '%14s | %s(%s): %s -> %s',
            ops_text, name, attr_text, inputs, outputs)

    # register the output tensor size
    for value, size in zip(node.outputs(), outputs):
        variables[value.unique()] = size

    # estimate of child nodes
    for block in node.blocks():
        raise NotImplementedError()

    return operations


def _get_default_estimators(
    strict: bool,


) -> Dict[str, Callable[
        [str, Dict[str, Any],
         List[Optional[List[int]]],
         List[Optional[List[int]]]],
        Tuple[int, List[Optional[List[int]]]]]]:
    if strict:
        estimate_unknown = _estimate_unknown_fatal
    else:
        estimate_unknown = _estimate_unknown_warning

    return collections.defaultdict(
        lambda: estimate_unknown,
        {
            'onnx::Conv': _estimate_convNd,
            'onnx::Gemm': _estimate_linear,
            'onnx::BatchNormalization': _estimate_batchnorm,
            'onnx::AveragePool': _estimate_avgpool,
            'onnx::MaxPool': _estimate_maxpool,
            'onnx::GlobalAveragePool': _estimate_globalavgpool,
            'onnx::GlobalMaxPool': _estimate_globalmaxpool,
            'onnx::Softmax': _estimate_softmax,
            'onnx::Transpose': _estimate_transpose,
            'onnx::Sigmoid': functools.partial(_estimate_operation_n, steps=4),
            'onnx::Relu': functools.partial(_estimate_operation_n, steps=2),
            'onnx::Add': functools.partial(_estimate_operation_n, steps=1),
            'onnx::Mul': functools.partial(_estimate_operation_n, steps=1),
            'onnx::Concat': _estimate_operation_zero,
            'onnx::Constant': _estimate_operation_zero,
            'onnx::Flatten': _estimate_operation_zero,
            'onnx::Gather': _estimate_operation_zero,
            'onnx::Reshape': _estimate_operation_zero,
            'onnx::Shape': _estimate_operation_zero,
            'onnx::Split': _estimate_operation_zero,
            'onnx::Squeeze': _estimate_operation_zero,
            'onnx::Unsqueeze': _estimate_operation_zero,
            'prim::ListConstruct': _estimate_operation_zero,
        }
    )


def _estimate_unknown_fatal(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    print(name)
    print(attrs)
    print(inputs)
    print(outputs)
    raise Exception(f'Unknown operation: {name}')


def _estimate_unknown_warning(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    LOGGER.warning('Unknown operation: %s', name)
    return 0, []


def _estimate_convNd(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[1] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of weight parameters is not provided.')

    # estimate the shape of the output tensor
    if outputs[0] is None:
        if inputs[0] is None:
            raise Exception(
                f'Failed at estimating `{name}`:'
                + ' the shape of the output tensor can not be estimated.')

        kernels = inputs[1][2:]
        pads = attrs['pads']
        strides = attrs['strides']
        dilations = attrs['dilations']
        feature_size = [sum(v) for v in zip(inputs[0][2:], pads[::2], pads[1::2])]
        panel_size = [(k - 1) * d + 1 for k, d in zip(kernels, dilations)]
        output_size = [inputs[0][0], inputs[1][0]] + [
            (f - p + 1) // s for f, p, s in zip(feature_size, panel_size, strides)]
    else:
        output_size = outputs[0]

    # count operations of convolutions
    in_channels = inputs[1][1]
    out_channels = inputs[1][0]
    kernels = inputs[1][2:]

    operations = in_channels * out_channels
    operations *= functools.reduce(operator.mul, kernels)

    # count operations of biases
    if len(inputs) == 3:
        operations += out_channels

    # estimate the number of operations
    groups = int(attrs['group'])

    operations *= groups
    operations *= functools.reduce(operator.mul, output_size[2:])
    operations *= output_size[0]

    return operations, [output_size]


def _estimate_linear(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[1] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of weight parameters is not provided.')

    if outputs[0] is None:
        if inputs[0] is None:
            raise Exception(
                f'Failed at estimating `{name}`:'
                + ' the shape of the output tensor can not be estimated.')

        output_size = [inputs[0][0], inputs[1][0]]
    else:
        output_size = outputs[0]

    operations = inputs[1][0] * inputs[1][1]

    if len(inputs) == 3:
        operations += inputs[1][0]

    return operations, [output_size]


def _estimate_batchnorm(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    operations = 2 * functools.reduce(operator.mul, inputs[0])

    return operations, [inputs[0]]


def _estimate_avgpool(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    if outputs[0] is None:
        kernels = attrs['kernel_shape']
        pads = attrs['pads']
        strides = attrs['strides']
        feature_size = [
            sum(v) for v in zip(inputs[0][2:], pads[::2], pads[1::2])]
        output_size = inputs[0][:2] + [
            (f - k + 1) // s for f, k, s in zip(feature_size, kernels, strides)]
    else:
        output_size = outputs[0]

    kernels = attrs['kernel_shape']
    operations = functools.reduce(operator.mul, kernels)
    operations *= functools.reduce(operator.mul, output_size)

    return operations, [output_size]


def _estimate_maxpool(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    if outputs[0] is None:
        kernels = attrs['kernel_shape']
        pads = attrs['pads']
        strides = attrs['strides']
        feature_size = [
            sum(v) for v in zip(inputs[0][2:], pads[::2], pads[1::2])]
        output_size = inputs[0][:2] + [
            (f - k + 1) // s for f, k, s in zip(feature_size, kernels, strides)]
    else:
        output_size = outputs[0]

    kernels = attrs['kernel_shape']
    operations = functools.reduce(operator.mul, kernels) - 1
    operations *= functools.reduce(operator.mul, output_size)

    return operations, [output_size]


def _estimate_globalavgpool(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    if outputs[0] is None:
        output_size = inputs[0][:2] + [1 for _ in inputs[0][2:]]
    else:
        output_size = outputs[0]

    operations = functools.reduce(operator.mul, inputs[0][2:])
    operations *= functools.reduce(operator.mul, inputs[0][:2])

    return operations, [output_size]


def _estimate_globalmaxpool(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    if outputs[0] is None:
        output_size = inputs[0][:2] + [1 for _ in inputs[0][2:]]
    else:
        output_size = outputs[0]

    operations = functools.reduce(operator.mul, inputs[0][2:]) - 1
    operations *= functools.reduce(operator.mul, inputs[0][:2])

    return operations, [output_size]


def _estimate_softmax(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    if outputs[0] is None:
        output_size = inputs[0]
    else:
        output_size = outputs[0]

    axis = attrs['axis']
    target = inputs[0][axis]
    others = [v for i, v in enumerate(inputs[0]) if i != axis]

    operations = target + (target - 1) + target
    operations *= functools.reduce(operator.mul, others)

    return operations, [output_size]


def _estimate_operation_n(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
    steps: int,
) -> Tuple[int, List[Optional[List[int]]]]:
    if inputs[0] is None:
        raise Exception(
            f'Failed at estimating `{name}`:'
            + ' the shape of input parameters is not provided.')

    if outputs[0] is None:
        input_sizes = [v for v in inputs if v is not None]
        input_length = max(len(v) for v in input_sizes)
        input_sizes = [[1] * (input_length - len(v)) + v for v in input_sizes]
        output_size = [max(v) for v in zip(*input_sizes)]
    else:
        output_size = outputs[0]

    operations = steps * functools.reduce(operator.mul, output_size)

    return operations, [output_size]


def _estimate_operation_zero(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:
    return 0, []


def _estimate_transpose(
    name: str,
    attrs: Dict[str, Any],
    inputs: List[Optional[List[int]]],
    outputs: List[Optional[List[int]]],
) -> Tuple[int, List[Optional[List[int]]]]:

    if outputs[0] is None:
        if inputs[0] is None:
            raise Exception(
                f'Failed at estimating `{name}`:'
                + ' the shape of input parameters is not provided.')

        output_size = [inputs[0][i] for i in attrs['perm']]
    else:
        output_size = outputs[0]

    return 0, [output_size]
