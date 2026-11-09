"""确认诊断器能区分前向、反向异常，并跳过尚未初始化的张量。"""
import json

import pytest
import torch

from scripts.diagnose_ucan_nonfinite import FirstNonfinite, replay


def test_finite_and_uninitialized_outputs():
    mode = FirstNonfinite(['finite'])
    with mode:
        torch.empty(8).fill_(1).square().sum()
    assert mode.report is None


def test_first_nonfinite_forward_operator():
    mode = FirstNonfinite(['fourier'])
    with pytest.raises(FloatingPointError), mode:
        torch.tensor([1.0]) / torch.tensor([0.0])
    assert mode.report['operator'] == 'aten.div.Tensor'
    assert mode.report['module_stack'] == ['fourier']
    assert mode.report['outputs'][0]['inf'] == 1
    assert all(item['inf'] == item['nan'] == 0 for item in mode.report['inputs'])


def test_finite_forward_with_nonfinite_backward():
    mode = FirstNonfinite([])
    with pytest.raises(FloatingPointError), mode:
        # 倒数前向有限，反向需要平方倒数，FP32 会发生溢出。
        torch.tensor([1e-20], requires_grad=True).reciprocal().sum().backward()
    assert mode.report['operator'] == 'aten.mul.Tensor'
    assert mode.report['outputs'][0]['inf'] == 1


def test_replay_records_backward_node_and_removes_hooks(tmp_path):
    class Reciprocal(torch.nn.Module):
        def forward(self, x):
            return x.reciprocal()

    net = Reciprocal()
    output = tmp_path / 'operator.json'
    replay(net, lambda pred, target: pred.sum(),
           torch.tensor([1e-20], requires_grad=True), torch.zeros(1), output)
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['operator'] == 'aten.mul.Tensor'
    assert report['module_stack'] == ['backward:ReciprocalBackward0']
    assert not net._forward_hooks and not net._forward_pre_hooks
