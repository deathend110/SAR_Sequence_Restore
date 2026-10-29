"""UCAN 的单帧 SAR 适配：复用训练流程和整帧分布式验证。"""
import torch
from torch.nn.parallel import DistributedDataParallel

from basicsr.utils.registry import MODEL_REGISTRY
from .distributed_validation import DistributedValidationMixin
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class UCANModel(DistributedValidationMixin, SRModel):

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        # 验证分片可能不等长或为空，前向绕过 DDP 的 buffer 同步。
        if isinstance(net, DistributedDataParallel):
            net = self.get_bare_model(net)
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                # UCAN 内部完成窗口 padding；保持与 SwinIR/HINT 相同的整帧评估协议。
                self.output = net(self.lq)
        finally:
            net.train(was_training)
