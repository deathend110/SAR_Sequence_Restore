"""SwinIR 验证前向的 padding、EMA 选择和模式恢复检查，无需 GPU。"""
import pytest
import torch

from basicsr.models.swinir_model import SwinIRModel


class RecordingNet(torch.nn.Module):

    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.input_shape = None

    def forward(self, image):
        self.input_shape = image.shape
        assert not self.training
        assert not torch.is_grad_enabled()
        return image + self.offset


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('ema', [True, False])
def test_inference_preserves_mode_and_crops_padding(training, ema):
    model = SwinIRModel.__new__(SwinIRModel)
    model.opt = {'network_g': {'window_size': 8}, 'scale': 1}
    model.net_g = RecordingNet(1).train(training)
    if ema:
        model.net_g_ema = RecordingNet(2).train(training)
    model.lq = torch.rand(1, 1, 13, 19)
    model.test()
    selected = model.net_g_ema if ema else model.net_g
    assert selected.input_shape == (1, 1, 16, 24)
    assert selected.training == training
    assert model.net_g.training == training
    if ema:
        assert model.net_g.input_shape is None
    torch.testing.assert_close(model.output, model.lq + (2 if ema else 1))
