import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch
from ..backbone.backbone import ResnetBackbone
from ..layers.decoder import Decoder
from ..layers.head import HeadPredictor
from ..loss_func.loss import Loss


class CenterNet(torch.nn.Module):

    r"""Implements CenterNet (object as points).
    Arguments:
        num_classes (int): number of objects
        backbone (string): the backbone to use (default: `resnet50`)
        cfg (configs): (default: None)
        device (string): cpu or cuda (default: None)
    """

    def __init__(self, num_classes, backbone='r50', cfg=None, device=None):
        super(CenterNet, self).__init__()

        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.backbone = ResnetBackbone(backbone=cfg.backbone if cfg else backbone, 
                                       pretrained=cfg.pretrained if cfg else False)
        self.upsample = Decoder(inplanes=self.backbone.outplanes,
                                bn_momentum=cfg.bn_momentum if cfg else 0.1,
                                deconv_with_bias=cfg.deconv_with_bias if cfg else False)
        self.predictor = HeadPredictor(num_classes=num_classes, channel=cfg.head_channel if cfg else 64)
        self.loss_func = Loss(cfg)
        self.post_processor = PostProcessor(topk=cfg.topk if cfg else 40, threshold=cfg.threshold if cfg else 0.1)
        self.to(self.device)

    def forward(self, inputs, targets=None):

        fmap = self.backbone(inputs)
        out = self.upsample(fmap)
        hm, wh, offset=self.predictor(out)
        if self.training:
            return self.loss_func(logits=[hm, wh, offset], targets=targets)

        return self.post_processor(hm.sigmoid(), wh, offset.tanh())

    @torch.no_grad()
    def detect_one_image(self, image_dir, threshold=0.2, classes=('circle', )):
        self.eval()

        rgb_image = cv2.imread(image_dir)[..., (2,1,0)]
        X, (image_h, image_w), *_ = preprocess_img(rgb_image)
                                                   
        detects = self(X.to(self.device))

        colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
        plt.figure(figsize=(12, 12))
        plt.imshow(rgb_image)
        currentAxis = plt.gca()
        for i, (boxes, scores, labels) in enumerate(detects):
            for box, score, label in zip(boxes, scores, labels): # [topk, 4], [topk], [topk]
                score, label = score.item(), label.long().item()
                if score < threshold:
                    continue
                label_name = classes[label]
                display_txt = '{}: {:.2f}'.format(label_name, score)
                color = colors[label]

                pt = box.detach().cpu().numpy() * (image_w, image_h,image_w, image_h)
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        plt.show()


class PostProcessor:
    def __init__(self, topk=100, threshold=0.1):
        self.topk = topk
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, hm_preds, wh_preds, offset_preds):

        batch_size, _, fmap_h, fmap_w = hm_preds.shape
        box_scale = torch.as_tensor([fmap_w, fmap_h, fmap_w, fmap_h], dtype=torch.float32, device=hm_preds.device)
        hm_preds = pool_nms(hm_preds) # [B, C, fmap_h, fmap_w] -> [B, C, fmap_h, fmap_w]

        scores, index, clses, ys, xs = topk_score(hm_preds, self.topk) # -> all shapes are [B, topk]
        reg = gather_feature(offset_preds, index, use_transform=True).reshape(batch_size, -1, 2) # [B, topK, 2] -> [B, topK, 2]
        wh = gather_feature(wh_preds, index, use_transform=True).reshape(batch_size, -1, 2) # [B, topK, 2] -> [B, topK, 2]

        xs = xs.unsqueeze(-1) + reg[..., :1] # -> [B, topK, 1]
        ys = ys.unsqueeze(-1) + reg[..., 1:] # -> [B, topK, 1]
        half_w = wh[..., :1] / 2
        half_h = wh[..., 1:] / 2
        batch_boxes = torch.cat([xs - half_w, ys - half_h, 
                                 xs + half_w, ys + half_h], dim=2) # -> [B, topK, 4]

        detects = []
        for boxes, score, label in zip(batch_boxes, scores, clses):
            mask = score.gt(self.threshold)
            instance_boxes = boxes[mask] / box_scale
            instance_scores = score[mask]
            instance_clses = label[mask]
            detects.append((instance_boxes, instance_scores, instance_clses))

        return detects


def pool_nms(hm, pool_size=3):
    padding = (pool_size - 1) // 2
    hm_max = torch.nn.functional.max_pool2d(hm, pool_size, stride=1, padding=padding)
    keep = (hm_max == hm).float()
    return hm * keep


def topk_score(scores, K):
    batch, channel, height, width = scores.shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K) # [B, C, HxW] -> [B, C, K], [B, C, K]

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float() # -> [B, C, K] 特征图映射到y轴
    topk_xs = (topk_inds % width).int().float()  # -> [B, C, K] 特征图映射到x轴

    # get all topk in in a batch
    topk_scores, index = torch.topk(topk_scores.reshape(batch, -1), K) # [B, C, K] -> [B, K], [B, K]
    # div by K because index is grouped by K, shape of (C, K)
    topk_clses = (index // K).int()
    topk_inds = topk_inds.reshape(batch, -1).gather(dim=1, index=index) # gather_feature(topk_inds.reshape(batch, -1, 1), index).reshape(batch, -1) # [B, CxK, 1] -> [B, K]
    topk_ys = topk_ys.reshape(batch, -1).gather(dim=1, index=index) # gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, -1) # [B, CxK, 1] -> [B, K]
    topk_xs = topk_xs.reshape(batch, -1).gather(dim=1, index=index) # gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, -1) # [B, CxK, 1] -> [B, K]

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feature(fmap, index, mask=None, use_transform=False):

    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel, *_ = fmap.shape
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(-1).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index) #  -> [B, K, dim]
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


# torch.gather https://blog.csdn.net/cpluss/article/details/90260550
# out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0 
# out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
# out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

def preprocess_img(image, input_size=512, pad=32, th_pad=22,
                   mean=(0.485, 0.456, 0.406), 
                   std=(0.229, 0.224, 0.225)):

    image_h, image_w, _ = image.shape
    if image_h < image_w:
        resize_w = input_size
        resize_h = int(image_h * input_size / image_w + 0.5)
        pad_pos, pad_neg = (input_size - resize_h) % pad, resize_h % pad
        _pad = pad_pos if pad_pos < th_pad else -pad_neg
        resize_h = resize_h + _pad
    else:
        resize_h = input_size
        resize_w = int(image_w * input_size / image_h + 0.5)
        pad_pos, pad_neg = (input_size - resize_w) % pad, resize_w % pad
        _pad = pad_pos if pad_pos < th_pad else -pad_neg
        resize_w = resize_w + _pad

    image=torch.as_tensor(
        (cv2.resize(image, (resize_w, resize_h)) / 255.0 - mean) / std, dtype=torch.float32
    ).permute(2,0,1).unsqueeze(0)
    return image, (image_h, image_w), (resize_h, resize_w)
