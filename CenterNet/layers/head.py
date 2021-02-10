import torch.nn as nn


class HeadPredictor(nn.Module):
    def __init__(self, num_classes, channel=64):
        super(HeadPredictor, self).__init__()

        self.cls_head = _make_head(256, channel, num_classes)
        self.wh_head = _make_head(256, channel, 2)
        self.reg_head = _make_head(256, channel, 2)

    def forward(self, x):
        hm = self.cls_head(x) #.sigmoid() # -> [B, C, 128, 128]
        wh = self.wh_head(x)              # -> [B, 2, 128, 128]
        offset = self.reg_head(x)         # -> [B, 2, 128, 128]
        return hm, wh, offset


def _make_head(in_channel, channel, out_channel, bias_fill=False, bias_value=0):
    in_conv = nn.Conv2d(in_channel, channel, kernel_size=3, padding=1)
    relu = nn.ReLU(inplace=True)
    out_conv = nn.Conv2d(channel, out_channel, kernel_size=1)
    if bias_fill:
        out_conv.bias.data.fill_(bias_value)
    return nn.Sequential(in_conv, relu, out_conv)
