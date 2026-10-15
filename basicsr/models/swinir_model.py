import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from .distributed_validation import DistributedValidationMixin


@MODEL_REGISTRY.register()
class SwinIRModel(DistributedValidationMixin, SRModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        # 各卡验证帧数可能不同，绕过 DDP 的前向 buffer 同步；训练仍使用原包装器。
        if isinstance(net, DistributedDataParallel):
            net = self.get_bare_model(net)
        was_training = net.training
        net.eval()
        with torch.no_grad():
            self.output = net(img)
        net.train(was_training)

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
