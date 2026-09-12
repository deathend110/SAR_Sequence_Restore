"""SwinIR 验证前向的 padding、EMA 选择和模式恢复检查，无需 GPU。"""
import pytest
import torch

from basicsr.archs.swinir_arch import SwinIR, SwinTransformerBlock
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


def test_dynamic_masks_are_shared_and_not_checkpointed(monkeypatch):
    net = SwinIR(img_size=8, in_chans=1, embed_dim=6, depths=[2, 2], num_heads=[2, 2],
                 window_size=4, upscale=1, upsampler='').eval()
    saved_state = {key: value.clone() for key, value in net.state_dict().items()}
    buffer_names = set(dict(net.named_buffers()))
    original = SwinTransformerBlock.calculate_mask
    calls = []

    def record(block, size, device=None):
        calls.append((size, device, block.shift_size))
        return original(block, size, device=device)

    monkeypatch.setattr(SwinTransformerBlock, 'calculate_mask', record)
    image = torch.rand(1, 1, 16, 24)
    with torch.no_grad():
        first = net(image)
        assert len(calls) == 2
        masks = dict(net._attn_masks)
        second = net(image)
        assert len(calls) == 2
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert all(net._attn_masks[key] is value for key, value in masks.items())
        net(torch.rand(1, 1, 24, 16))
        assert len(calls) == 4
        assert all(net._attn_masks[key] is not value for key, value in masks.items())
        net(torch.rand(1, 1, 8, 8))
        assert len(calls) == 4 and not net._attn_masks
    assert set(net.state_dict()) == set(saved_state)
    assert set(dict(net.named_buffers())) == buffer_names
    net.load_state_dict(saved_state, strict=True)


def test_checkpoint_forward_preserves_gradients():
    args = dict(img_size=8, in_chans=1, embed_dim=6, depths=[2], num_heads=[2],
                window_size=4, upscale=1, upsampler='')
    direct = SwinIR(**args)
    checkpointed = SwinIR(**args, use_checkpoint=True)
    checkpointed.load_state_dict(direct.state_dict())
    image = torch.rand(1, 1, 16, 16)
    for net in (direct, checkpointed):
        torch.manual_seed(42)
        net(image).square().mean().backward()
    for expected, actual in zip(direct.parameters(), checkpointed.parameters()):
        # 重计算的梯度累加顺序不同，使用 FP32 容差；恢复输出另作严格逐元素对照。
        torch.testing.assert_close(actual.grad, expected.grad)
