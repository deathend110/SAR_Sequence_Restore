"""UCAN 九帧序列训练与验证模型包装（支持从 A_init 加载骨干或完整配对初始化权重）。"""

import torch
from torch.nn.parallel import DistributedDataParallel

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .sequence_validation import SequenceDistributedValidationMixin
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class UCANSequenceModel(SequenceDistributedValidationMixin, SRModel):
    """九帧联合恢复模型：全部九帧等权 Charbonnier 损失、支持骨干热启动与配对恢复。"""

    def __init__(self, opt):
        super().__init__(opt)
        # 若使用 train_frame_chunk_size 对共享骨干重复执行多次 checkpoint，开启 static_graph 防止 DDP 报 ready twice
        if self.opt.get('dist') and isinstance(self.net_g, DistributedDataParallel):
            bare = self.get_bare_model(self.net_g)
            if getattr(bare, 'train_frame_chunk_size', 0) > 0:
                self.net_g._set_static_graph()
        backbone_path = self.opt['path'].get('pretrain_backbone_g')
        full_path = self.opt['path'].get('pretrain_network_g')
        if backbone_path is not None and full_path is None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            ckpt = torch.load(backbone_path, map_location='cpu', weights_only=True)
            state = ckpt[param_key] if (param_key is not None and param_key in ckpt) else ckpt
            bare = self.get_bare_model(self.net_g)
            bare.load_backbone_state_dict(state, strict=True)
            logger = get_root_logger()
            logger.info(f'Loaded A_init backbone into UCANSequence from {backbone_path}.')

    def feed_data(self, data):
        self.lq = data['lq_seq'].to(self.device)  # [B, 9, 1, H, W]
        if 'gt_seq' in data:
            self.gt = data['gt_seq'].to(self.device)
        self.frame_ids = data['frame_ids'].to(self.device) if 'frame_ids' in data else None
        self.token_quality = data['token_quality'].to(self.device) if 'token_quality' in data else None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, frame_ids=self.frame_ids, token_quality=self.token_quality)

        # 将 [B, 9, 1, H, W] 展平为 [B*9, 1, H, W] 后使用等权 Charbonnier 损失
        pred_flat = self.output.reshape(-1, *self.output.shape[2:])
        gt_flat = self.gt.reshape(-1, *self.gt.shape[2:])
        l_total = 0
        loss_dict = {}
        if self.cri_pix:
            l_pix = self.cri_pix(pred_flat, gt_flat)
            l_total = l_total + l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        if isinstance(net, DistributedDataParallel):
            net = self.get_bare_model(net)
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                self.output = self._run_sequence_inference(
                    net, self.lq, frame_ids=self.frame_ids, token_quality=self.token_quality
                )
        finally:
            net.train(was_training)
