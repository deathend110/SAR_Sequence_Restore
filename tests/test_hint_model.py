"""HINT 推理模式与持久缓存检查，无需 GPU。"""
import pytest
import torch

from basicsr.models.hint_model import HINTModel


class StatefulNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return image + self.calls


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('ema', [True, False])
def test_hint_inference_preserves_mode_and_state(training, ema):
    model = HINTModel.__new__(HINTModel)
    model.net_g = StatefulNet().train(training)
    if ema:
        model.net_g_ema = StatefulNet().train(training)
    model.lq = torch.rand(1, 1, 16, 16)
    selected = model.net_g_ema if ema else model.net_g
    for count in (1, 2):
        model.test()
        torch.testing.assert_close(model.output, model.lq + count)
        assert selected.training == training
        assert selected.calls == count
    if ema:
        assert model.net_g.calls == 0
