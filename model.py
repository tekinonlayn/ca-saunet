import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        y = torch.cat([x_h.permute(0, 1, 3, 2), x_w], dim=3)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y_h, y_w = torch.split(y, [h, w], dim=3)
        y_h = y_h.permute(0, 1, 3, 2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w.expand_as(x)

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_features, attn_features, up_factor=1, normalize_attn=False):
        super(SpatialAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.down = nn.Conv2d(in_features, attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(attn_features, 1, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(attn_features)

    def forward(self, x):
        b, c, h, w = x.size()
        attn = self.down(x)
        attn = self.relu(self.bn(attn))
        attn = self.phi(attn)
        if self.normalize_attn:
            attn = F.softmax(attn.view(b, 1, -1), dim=2).view(b, 1, h, w)
        else:
            attn = torch.sigmoid(attn)
        return attn

class _MRF(nn.Module):
    def __init__(self, inchannels):
        super(_MRF, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(inchannels[0], inchannels[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(inchannels[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, channels):
        if channels[0].shape[2:] == channels[1].shape[2:]:
            return torch.cat([channels[0], channels[1]], dim=1)
        else:
            return torch.cat([self.up(channels[0]), channels[1]], dim=1)

class DualAttBlock(nn.Module):
    def __init__(self, inchannels=[128, 256], outchannels=256):
        super(DualAttBlock, self).__init__()
        inchs = sum(inchannels)
        self.mrf = _MRF(inchannels)
        self.spatialAttn = SpatialAttentionBlock(outchannels, outchannels // 4, up_factor=2)
        self.channelAttn = CoordinateAttention(outchannels, reduction=16)
        self.c3x3rb = nn.Sequential(
            nn.Conv2d(inchs, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        fused = self.mrf(x)
        fused = self.c3x3rb(fused)
        spatial = self.spatialAttn(fused)
        channel = self.channelAttn(fused)
        out = (spatial.expand_as(channel) + 1) * channel
        return out, spatial

class Norm2d(nn.Module):
    def __init__(self, num_features):
        super(Norm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class GatedSpatialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(GatedSpatialConv2d, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self._gate_conv = nn.Sequential(
            Norm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            Norm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        input_features = input_features * (alphas + 1)
        return self._conv(input_features), alphas

    def reset_parameters(self):
        nn.init.xavier_normal_(self._conv.weight)
        if self._conv.bias is not None:
            nn.init.zeros_(self._conv.bias)

class CA_SAUNet(nn.Module):
    def __init__(self, num_classes=2, num_filters=32, pretrained=False, is_deconv=True, dataset='plantdoc'):
        super(CA_SAUNet, self).__init__()
        self.num_classes = num_classes
        self.dataset = dataset
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        if dataset == 'plantdoc':
            self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.encoder3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.encoder4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.bottleneck = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
            self.reduce_bottleneck = nn.Conv2d(1024, 512, kernel_size=1)
        else:
            self.encoder = torchvision.models.densenet121(pretrained=pretrained)
            self.conv1 = nn.Sequential(self.encoder.features.conv0, self.encoder.features.norm0)
            self.conv2 = self.encoder.features.denseblock1
            self.conv2t = self.encoder.features.transition1
            self.conv3 = self.encoder.features.denseblock2
            self.conv3t = self.encoder.features.transition2
            self.conv4 = self.encoder.features.denseblock3
            self.conv4t = self.encoder.features.transition3
            self.conv5 = nn.Sequential(self.encoder.features.denseblock4, self.encoder.features.norm5)
            self.c3 = nn.Conv2d(256, 1, kernel_size=1)
            self.c4 = nn.Conv2d(512, 1, kernel_size=1)
            self.c5 = nn.Conv2d(1024, 1, kernel_size=1)
            self.d0 = nn.Conv2d(128, 64, kernel_size=1)
            self.res1 = ResBlock(64, 64)
            self.d1 = nn.Conv2d(64, 32, kernel_size=1)
            self.res2 = ResBlock(32, 32)
            self.d2 = nn.Conv2d(32, 16, kernel_size=1)
            self.res3 = ResBlock(16, 16)
            self.d3 = nn.Conv2d(16, 8, kernel_size=1)
            self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
            self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
            self.gate1 = GatedSpatialConv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.gate2 = GatedSpatialConv2d(16, 16, kernel_size=3, stride=1, padding=1)
            self.gate3 = GatedSpatialConv2d(8, 8, kernel_size=3, stride=1, padding=1)
            self.expand = nn.Sequential(
                nn.Conv2d(1, num_filters, kernel_size=1),
                Norm2d(num_filters),
                nn.ReLU(inplace=True)
            )

        self.center = nn.Sequential(
            nn.Conv2d(1024, num_filters * 8 * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 8 * 2),
            nn.ReLU(inplace=True)
        )
        self.dec5 = DualAttBlock(inchannels=[512, 1024], outchannels=512)
        self.dec4 = DualAttBlock(inchannels=[512, 512], outchannels=256)
        self.dec3 = DualAttBlock(inchannels=[256, 256], outchannels=128)
        self.dec2 = DualAttBlock(inchannels=[128, 128], outchannels=64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(num_filters + 64, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x, return_att=False):
        x_size = x.size()

        if self.dataset == 'plantdoc':
            e1 = self.encoder1(x)
            e2 = self.encoder2(self.pool(e1))
            e3 = self.encoder3(self.pool(e2))
            e4 = self.encoder4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            b_reduced = self.reduce_bottleneck(b)
            d5_up = F.interpolate(b_reduced, scale_factor=2, mode='bilinear', align_corners=True)
            d5, att5 = self.dec5([d5_up, e4])
        else:
            conv1 = self.conv1(x)
            conv2 = self.conv2t(self.conv2(conv1))
            conv3 = self.conv3t(self.conv3(conv2))
            conv4 = self.conv4t(self.conv4(conv3))
            conv5 = self.conv5(conv4)
            ss = F.interpolate(self.d0(conv2), x_size[2:], mode='bilinear', align_corners=True)
            ss = self.res1(ss)
            c3 = F.interpolate(self.c3(conv3), x_size[2:], mode='bilinear', align_corners=True)
            ss, g1 = self.gate1(ss, c3)
            ss = self.res2(ss)
            ss = self.d2(ss)
            c4 = F.interpolate(self.c4(conv4), x_size[2:], mode='bilinear', align_corners=True)
            ss, g2 = self.gate2(ss, c4)
            ss = self.res3(ss)
            ss = self.d3(ss)
            c5 = F.interpolate(self.c5(conv5), x_size[2:], mode='bilinear', align_corners=True)
            ss, g3 = self.gate3(ss, c5)
            ss = self.fuse(ss)
            ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
            edge_out = self.sigmoid(ss)
            im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
            canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]), dtype=np.uint8)
            for i in range(x_size[0]):
                canny[i] = cv2.Canny(im_arr[i], 10, 100)
            canny = torch.from_numpy(canny).float().to(x.device)
            cat = torch.cat([edge_out, canny], dim=1)
            acts = self.cw(cat)
            acts = self.sigmoid(acts)
            edge = self.expand(acts)
            center = self.center(self.pool(conv5))
            d5, att5 = self.dec5([center, conv5])
            conv4_up = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)
            d5 = d5

        d4, att4 = self.dec4([d5, conv4_up if self.dataset == 'isic' else e4])
        conv3_up = F.interpolate(conv3 if self.dataset == 'isic' else e3, scale_factor=2, mode='bilinear', align_corners=True)
        d3, att3 = self.dec3([d4, conv3_up])
        conv2_up = F.interpolate(conv2 if self.dataset == 'isic' else e2, scale_factor=2, mode='bilinear', align_corners=True)
        d2, att2 = self.dec2([d3, conv2_up])
        d2_up = F.interpolate(d2, size=(conv1 if self.dataset == 'isic' else e1).shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d2_up, conv1 if self.dataset == 'isic' else e1], dim=1)
        d1 = self.dec1(d1)
        d1_up = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.dataset == 'isic':
            d0 = torch.cat([d1_up, edge], dim=1)
        else:
            d0 = d1_up

        dec0 = self.dec0(d0)
        x_out = self.final(dec0)

        att2 = F.interpolate(att2, scale_factor=4, mode='bilinear', align_corners=True)
        att3 = F.interpolate(att3, scale_factor=8, mode='bilinear', align_corners=True)
        att4 = F.interpolate(att4, scale_factor=16, mode='bilinear', align_corners=True)
        att5 = F.interpolate(att5, scale_factor=32, mode='bilinear', align_corners=True)

        if return_att:
            if self.dataset == 'isic':
                return x_out, edge_out, [att2, att3, att4, att5, g1, g2, g3]
            return x_out, [att2, att3, att4, att5]
        return x_out, edge_out if self.dataset == 'isic' else None