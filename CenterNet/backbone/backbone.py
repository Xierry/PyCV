import torch
import torchvision.models.resnet as resnet


class ResnetBackbone(torch.nn.Module):
    def __init__(self, backbone='resnet50', pretrained=False):
        super().__init__()

        self.encoder0, self.encoder1, self.encoder2, \
        self.encoder3, self.encoder4, self.fpn, self.outplanes = \
            _make_backbone(backbone, pretrained=pretrained)

    def forward(self, x):

        enc0 = self.encoder0(x)     # -> [B, 64, 128, 128]
        enc1 = self.encoder1(enc0)  # -> [B, 64, 128, 128] or [B, 256, 128, 128]
        enc2 = self.encoder2(enc1)  # -> [B, 128, 64, 64]  or [B, 512, 64, 64]
        enc3 = self.encoder3(enc2)  # -> [B, 256, 32, 32]  or [B, 1024, 32, 32]
        enc4 = self.encoder4(enc3)  # -> [B, 512, 16, 16]  or [B, 2048, 16, 16]

        return self.fpn(enc1, enc2, enc3, enc4)


class FPN(torch.nn.Module):
    def __init__(self, inplanes):
        super(FPN, self).__init__()

        self.laterals = torch.nn.ModuleList([Conv1x1(inplanes * (2 ** c), inplanes * (2 ** max(c-1, 0))) 
                                             for c in range(4)])
        self.smooths = torch.nn.ModuleList([Conv3x3(inplanes * (2 ** c), inplanes * (2 ** c)) 
                                            for c in range(1, 4)])
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, *features):

        fmap3 = self.laterals[3](features[3]) # [B, 2048, 16, 16] -> [2, 1024, 16, 16]
        fmap2 = self.laterals[2](features[2] + torch.nn.functional.interpolate(
                                 fmap3, scale_factor=2, mode="nearest")) # [B, 1024, 32, 32] -> [B, 512, 32, 32]
        fmap1 = self.laterals[1](features[1] + torch.nn.functional.interpolate(
                                 fmap2, scale_factor=2, mode="nearest")) # [B, 512, 64, 64] -> [B, 256, 64, 64]
        fmap0 = self.laterals[0](features[0] + torch.nn.functional.interpolate(
                                 fmap1, scale_factor=2, mode="nearest")) # [B, 256, 128, 128] -> [B, 256, 128, 128]
        fmap1 = self.smooths[0](torch.cat([fmap1, self.pooling(fmap0)], dim=1)) # [B, 256, 64, 64] -> [B, 512, 64, 64]
        fmap2 = self.smooths[1](torch.cat([fmap2, self.pooling(fmap1)], dim=1)) # [B, 512, 32, 32] -> [B, 1024, 32, 32]
        fmap3 = self.smooths[2](torch.cat([fmap3, self.pooling(fmap2)], dim=1)) # [B, 1024, 16, 16]-> [B, 2048, 16, 16]

        return fmap3


class Conv1x1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0):
        super().__init__()
        conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        norm = torch.nn.BatchNorm2d(out_channels)
        active = torch.nn.ReLU(True) 
        if negative_slope > 0:
            active = torch.nn.LeakyReLU(negative_slope, True)
        self.block = torch.nn.Sequential(conv, norm, active)

    def forward(self, x):
        return self.block(x)


class Conv3x3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0):
        super().__init__()
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        norm = torch.nn.BatchNorm2d(out_channels)
        active = torch.nn.ReLU(True)
        if negative_slope > 0:
            active = torch.nn.LeakyReLU(negative_slope, True)
        self.block = torch.nn.Sequential(conv, norm, active)

    def forward(self, x):
        return self.block(x)


def _make_backbone(name='r50', inplanes_backbone=256, pretrained=False):

    if name in ('r50', 'resnet50'):
        backbone = resnet.resnet50(pretrained=pretrained)
    elif name in ('r101', 'resnet101'):
        backbone = resnet.resnet101(pretrained=pretrained)
    elif name in ('r152', 'resnet152'):
        backbone = resnet.resnet152(pretrained=pretrained)
    elif name in ('rx50', 'resnext50_32x4d'):
        backbone = resnet.resnext50_32x4d(pretrained=pretrained)
    elif name in ('rx101', 'resnext101_32x8d'):
        backbone = resnet.resnext101_32x8d(pretrained=pretrained)
    # elif name in ('r50d', 'gluon_resnet50_v1d'):
    #     net = timm.create_model('gluon_resnet50_v1d', pretrained=pretrained)
    #     convert_to_inplace_relu(net)
    # elif name in ('r101d', 'gluon_resnet101_v1d'):
    #     net = timm.create_model('gluon_resnet101_v1d', pretrained=pretrained)
    #     convert_to_inplace_relu(net)     
    else:
        inplanes_backbone //= 4
        if name in ('r18', 'resnet18'):
            backbone = resnet.resnet18(pretrained=pretrained)
        elif name in ('r34', 'resnet34'):
            backbone = resnet.resnet34(pretrained=pretrained)
        else:
            assert False, "No BackBone: {}".format(name)
    
    encoder0 = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
    encoder1 = backbone.layer1
    encoder2 = backbone.layer2
    encoder3 = backbone.layer3
    encoder4 = backbone.layer4
    fpn=FPN(inplanes=inplanes_backbone)

    return encoder0, encoder1, encoder2, encoder3, encoder4, fpn, inplanes_backbone * (2 ** 3)


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            m.inplace = True
