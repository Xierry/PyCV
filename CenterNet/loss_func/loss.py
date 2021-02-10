import torch.nn.functional as F


class Loss:
    def __init__(self, cfg=None):
        super(Loss, self).__init__()
        self.alpha = cfg.loss_alpha if cfg else 1.0
        self.beta = cfg.loss_beta if cfg else 0.2
        self.gamma = cfg.loss_gamma if cfg else 1.0

    def __call__(self, logits, targets):

        pred_hm, pred_wh, pred_offset = logits   # -> [B, C, 120, 128], [B, 2, 120, 128], [B, 2, 120, 128]
        *_, gt_hm, infos = targets

        cls_loss = modified_focal_loss(pred_hm, gt_hm)

        num_objs = sum(len(info['ct']) for info in infos)

        wh_losses, offset_losses = zip(*[l1_loss(*args) for args in zip(pred_wh, pred_offset, infos)])

        reg_loss = self.beta * sum(wh_losses) + self.gamma * sum(offset_losses)

        return cls_loss * self.alpha, reg_loss / (num_objs + 1e-6)


def l1_loss(pred_wh, pred_offset, info):

    device = pred_wh.device
    ct = info['ct'].to(device)  # 特征图的GT中心点
    ct_int = ct.add(0.5).floor().long() # 四舍五入到最近的对应整数特征图

    pos_pred_wh = pred_wh[..., ct_int[1], ct_int[0]].view(-1)
    pos_pred_offset = pred_offset[..., ct_int[1], ct_int[0]].view(-1)

    wh = info['wh'].reshape(-1).to(device)
    offset = (ct - ct_int.float()).atanh().reshape(-1).to(device)

    wh_loss = F.l1_loss(pos_pred_wh, wh, reduction='sum')
    offset_loss = F.l1_loss(pos_pred_offset, offset, reduction='sum')

    return wh_loss, offset_loss


def modified_focal_loss(logits, targets):
    '''
    Modified focal loss. the same as CornerNet.
      Arguments:
        logits: BCHW hm
        targets: BCHW gt_hm
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = (1 - targets).pow(4)
    
    pred_sigmoid = logits.sigmoid()
    pos_loss = - F.logsigmoid(logits) * (1 - pred_sigmoid).pow(2) * pos_inds
    neg_loss = logits.exp().add(1.0).log() * pred_sigmoid.pow(2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = neg_loss
    else:
        loss = (pos_loss + neg_loss) / num_pos
    return loss
