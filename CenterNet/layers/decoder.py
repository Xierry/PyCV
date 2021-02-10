from torch import nn


class Decoder(nn.Module):
    def __init__(self, inplanes, num_filters=(256, 256, 256), num_kernels=(4, 4, 4), bn_momentum=0.1, deconv_with_bias = False):
        super(Decoder, self).__init__()
        # backbone output: [b, 2048, _h, _w]
        self.decoder = _make_deconv_layer(inplanes, num_filters, num_kernels, bn_momentum, deconv_with_bias)

    def forward(self, x):
        return self.decoder(x)


def _make_deconv_layer(inplanes, num_filters, num_kernels, bn_momentum, deconv_with_bias):
    layers = []
    for outplanes, kernel in zip(num_filters, num_kernels):
        padding = 0 if kernel == 2 else 1
        output_padding = 1 if kernel == 3 else 0
        layers.append(
            nn.ConvTranspose2d(
                in_channels=inplanes,
                out_channels=outplanes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=deconv_with_bias))
        layers.append(nn.BatchNorm2d(outplanes, momentum=bn_momentum))
        layers.append(nn.ReLU(inplace=True))
        inplanes = outplanes
    return nn.Sequential(*layers)
