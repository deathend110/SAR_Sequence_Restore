"""HINT 的 DDP 验证适配，保留网络原有持久缓存生命周期。"""
import torch
from torch.nn.parallel import DistributedDataParallel

from basicsr.utils.registry import MODEL_REGISTRY
from .distributed_validation import DistributedValidationMixin
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class HINTModel(DistributedValidationMixin, SRModel):

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        # 分片可能不等长或为空，验证前向绕过 DDP 的 buffer 同步。
        if isinstance(net, DistributedDataParallel):
            net = self.get_bare_model(net)
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                self.output = net(self.lq)
        finally:
            net.train(was_training)
