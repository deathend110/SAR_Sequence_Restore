"""验证 SWT 的方向、边界、数值尺度和梯度；全部使用 CPU。"""

import numpy as np
import pytest
import torch

from basicsr.losses import build_loss
from basicsr.losses.wavelet_loss import SWTLoss


@pytest.mark.parametrize('shape', [(1, 1, 40, 42), (2, 3, 39, 41)])
def test_identity_and_channelwise_transform(shape):
    image = torch.randn(shape, dtype=torch.float64)
    criterion = build_loss({'type': 'SWTLoss'})
    assert criterion(image, image).item() == 0
    bands = criterion.transform(image)
    assert bands.shape == (shape[0], shape[1], 4, shape[2], shape[3])
    for channel in range(shape[1]):
        torch.testing.assert_close(bands[:, channel:channel + 1], criterion.transform(image[:, channel:channel + 1]))
    assert not list(criterion.parameters())


def test_haar_subbands_have_expected_directions_and_normalization():
    image = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    a = image
    b = image.roll(-1, dims=3)
    c = image.roll(-1, dims=2)
    d = image.roll((-1, -1), dims=(2, 3))
    expected = torch.stack(((a + b + c + d) / 4, (a + b - c - d) / (2 * np.sqrt(2)), (a - b + c - d) / (2 * np.sqrt(2)),
                            (a - b - c + d) / 2),
                           dim=2)
    torch.testing.assert_close(SWTLoss(wavelet='haar', mode='periodic').transform(image), expected)


@pytest.mark.parametrize('wavelet', ['haar', 'sym19'])
def test_constant_error_only_contributes_to_low_frequency(wavelet):
    target = torch.zeros(1, 1, 40, 42, dtype=torch.float64)
    criterion = SWTLoss(wavelet=wavelet, loss_weight_ll=0.3, loss_weight=2)
    torch.testing.assert_close(criterion(target + 0.2, target), torch.tensor(0.12, dtype=torch.float64))
    high_only = SWTLoss(wavelet=wavelet, loss_weight_ll=0)
    torch.testing.assert_close(
        high_only(target + 0.2, target), torch.tensor(0., dtype=torch.float64), atol=1e-12, rtol=0)


def test_reductions_equal_separate_prediction_and_target_transforms():
    generator = torch.Generator().manual_seed(42)
    pred = torch.randn(2, 1, 39, 41, dtype=torch.float64, generator=generator)
    target = torch.randn(pred.shape, dtype=pred.dtype, generator=generator)
    options = dict(loss_weight=1.7, loss_weight_ll=0.2, loss_weight_lh=0.3, loss_weight_hl=0.4, loss_weight_hh=0.5)
    criterion = SWTLoss(**options, reduction='none')
    difference = (criterion.transform(pred) - criterion.transform(target)).abs()
    expected = (difference * torch.tensor([0.2, 0.3, 0.4, 0.5]).to(difference).view(1, 1, 4, 1, 1)).sum(2) * 1.7
    torch.testing.assert_close(criterion(pred, target), expected)
    torch.testing.assert_close(SWTLoss(**options)(pred, target), expected.mean())
    torch.testing.assert_close(SWTLoss(**options, reduction='sum')(pred, target), expected.sum())


def test_periodic_transform_is_equivariant_to_joint_shifts():
    image = torch.randn(1, 1, 23, 25, dtype=torch.float64)
    criterion = SWTLoss(mode='periodic')
    shifted = criterion.transform(image.roll((3, -5), dims=(-2, -1)))
    torch.testing.assert_close(shifted, criterion.transform(image).roll((3, -5), dims=(-2, -1)))


def test_backward_matches_numerical_derivatives():
    generator = torch.Generator().manual_seed(42)
    pred = torch.randn(1, 1, 3, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    target = torch.randn(pred.shape, dtype=pred.dtype, generator=generator)
    assert torch.autograd.gradcheck(SWTLoss(mode='periodic'), (pred, target), fast_mode=True)
    value = SWTLoss(mode='periodic')(pred, target)
    value.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_float32_backward_on_sar_crop():
    generator = torch.Generator().manual_seed(42)
    pred = torch.rand(2, 1, 128, 128, generator=generator, requires_grad=True)
    target = torch.rand(pred.shape, generator=generator)
    value = SWTLoss()(pred, target)
    assert value.dtype == pred.dtype == torch.float32
    value.backward()
    assert torch.isfinite(value) and value > 0
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


@pytest.mark.parametrize('options', [
    dict(wavelet='unknown'),
    dict(mode='unknown'),
    dict(reduction='unknown'),
    dict(loss_weight=-1),
    dict(loss_weight_hh=float('nan'))
])
def test_reject_invalid_options(options):
    with pytest.raises(ValueError):
        SWTLoss(**options)


def test_reject_shape_broadcasting_and_small_reflection_inputs():
    pred = torch.zeros(1, 1, 40, 40)
    with pytest.raises(ValueError, match='identical shapes'):
        SWTLoss()(pred, pred[:, :, :1])
    with pytest.raises(ValueError, match='same dtype'):
        SWTLoss()(pred, pred.double())
    with pytest.raises(ValueError, match='dimension > 19'):
        SWTLoss().transform(pred[:, :, :19])


def test_sym19_filters_match_optional_pywavelets_reference():
    # PyWavelets 仅用于独立参考验证，生产实现和其余测试均不依赖该包。
    pywt = pytest.importorskip('pywt')
    wave = pywt.Wavelet('sym19')
    criterion = SWTLoss(mode='periodic')
    expected = np.stack((np.asarray(wave.dec_lo) / sum(wave.dec_lo), wave.dec_hi))[:, ::-1].copy()
    np.testing.assert_array_equal(criterion.row_filters[:, 0, 0].numpy(), expected)
    image = np.random.default_rng(42).normal(size=(40, 42))
    custom = pywt.Wavelet('ucan_normalized', [
        np.asarray(wave.dec_lo) / sum(wave.dec_lo), wave.dec_hi, 2 * np.asarray(wave.rec_lo) / sum(wave.rec_lo),
        wave.rec_hi
    ])
    ll, (lh, hl, hh) = pywt.swt2(image, custom, level=1)[0]
    expected_bands = np.stack((ll, lh, hl, hh))
    actual = criterion.transform(torch.from_numpy(image)[None, None])[0, 0].numpy()
    np.testing.assert_allclose(actual, expected_bands, rtol=1e-12, atol=1e-12)
