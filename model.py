from vit_seg_modeling import Transformer
import torch
from torch import nn
import torch.nn.functional as F
from vit_seg_modeling import Transformer, CONFIGS

resnet_channels = [64, 256, 512, 768]


def norm_layer(channel, norm_name='gn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


def Increasing_Dimension(in_):
    out_ = in_.view(in_.size(0) // 5, 5, in_.size(1), in_.size(2), in_.size(3)).permute(0, 2, 1, 3, 4)
    return out_


def Reducing_Dimension(in_):
    in_ = in_.permute(0, 2, 1, 3, 4)
    out_ = in_.contiguous().view(in_.size(0) * in_.size(1), in_.size(2), in_.size(3), in_.size(4))
    return out_


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, norm=True, relu=True):
        super(BasicConv2d, self).__init__()
        self.norm = norm
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        if norm:
            self.bn_layer = norm_layer(out_planes)
        if relu:
            self.relu_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn_layer(x)
        if self.relu:
            x = self.relu_layer(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(out_channel, out_channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(out_channel // 4, out_channel, bias=False), nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
                                  norm_layer(out_channel),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                   norm_layer(out_channel))

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv(x)
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.out_channel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        return F.relu(y + self.conv2(y))


class UBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(UBlock, self).__init__()
        convs = []
        convs.append(nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 2, 1, bias=False),
            norm_layer(channels)
        ))
        for i in range(2, 4):
            convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                norm_layer(channels)
            ))
        convs.append(nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False),
            norm_layer(channels)
        ))

        convs2 = []
        for i in range(1, 4):
            convs2.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                norm_layer(channels)
            ))
        self.convs = nn.ModuleList(convs)
        self.convs2 = nn.ModuleList(convs2)
        self.final = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            norm_layer(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = []
        target_shape = x.shape[2:]
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            res.append(x)

        for i in range(len(res) - 1, 0, -1):
            res[i] = F.interpolate(res[i], res[i - 1].shape[2:], mode='bilinear', align_corners=True)
            res[i - 1] = res[i] + res[i - 1]
            res[i - 1] = self.convs2[i - 1](res[i - 1])
        return self.final(F.interpolate(res[0], target_shape, mode='bilinear', align_corners=True))


# Temporal contextual module
class TCM(nn.Module):
    def __init__(self, channels):
        super(TCM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0), dilation=(1, 1, 1)),
            norm_layer(channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0), dilation=(2, 1, 1)),
            norm_layer(channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0), dilation=(3, 1, 1)),
            norm_layer(channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0), dilation=(4, 1, 1)),
            norm_layer(channels)
        )
        self.reduce = nn.Sequential(
            nn.Conv3d(channels, channels, (4, 1, 1), stride=1, padding=0), norm_layer(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            norm_layer(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            norm_layer(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = Increasing_Dimension(x)
        x1 = x1.repeat(1, 1, 3, 1, 1)
        out1 = self.conv1(x1[:, :, 4:11, :, :])
        out2 = self.conv2(x1[:, :, 3:12, :, :])
        out3 = self.conv3(x1[:, :, 2:13, :, :])
        out4 = self.conv4(x1[:, :, 1:14, :, :])

        out1 = Reducing_Dimension(out1).unsqueeze(2)
        out2 = Reducing_Dimension(out2).unsqueeze(2)
        out3 = Reducing_Dimension(out3).unsqueeze(2)
        out4 = Reducing_Dimension(out4).unsqueeze(2)

        out_cat = torch.cat([out1, out2, out3, out4], dim=2)
        out_cat = self.reduce(out_cat).squeeze()
        # out_cat = Reducing_Dimension(out_cat)

        # 2021.08.11
        # out_cat = self.conv_cat(out_cat)
        return self.relu(x + out_cat)

        # x = x + x * torch.sigmoid(out_cat)
        # return self.relu(x + self.conv_cat(x))


# Spatial contextual module
class SCM(nn.Module):
    def __init__(self, channels):
        super(SCM, self).__init__()
        self.ublock = UBlock(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.ublock(x))


class FusionD(nn.Module):
    def __init__(self, config):
        super(FusionD, self).__init__()
        self.config = config
        self.relu = nn.ReLU()
        self.fuse_high = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False), nn.Sigmoid()
        )
        self.fuse_middle = nn.Sequential(
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False), norm_layer(config.channels)
        )
        self.fuse_cat = nn.Sequential(
            nn.Conv2d(config.channels * 2, config.channels, 3, 1, 1, bias=False), norm_layer(config.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.channels, config.channels, 3, 1, 1, bias=False), norm_layer(config.channels),
        )
        self.scm = SCM(channels=config.channels)

    def forward(self, low, middle, high=None):
        if high is not None:
            high = F.interpolate(high, low.shape[2:], mode='bilinear', align_corners=True)
        middle = F.interpolate(middle, low.shape[2:], mode='bilinear', align_corners=True)
        middle = self.fuse_middle(middle) + low
        if high is not None:
            high = self.fuse_high(high) * low
            middle_high_cat = torch.cat([middle, high], dim=1)
            fused = self.fuse_cat(middle_high_cat)
        else:
            fused = middle
        fused = self.relu(fused)
        fused = self.scm(fused)

        return fused


class FinetuneNet(nn.Module):
    def __init__(self, config):
        super(FinetuneNet, self).__init__()
        self.config = config
        self.encoder = Transformer(config=CONFIGS['R50-ViT-B_16'], img_size=config.trainsize, vis=False)

        compress = []
        for i in range(4):
            compress.append(ChannelAttention(in_channel=resnet_channels[i], out_channel=config.channels))
        self.compress = nn.ModuleList(compress)

        fusion = []
        for i in range(3):
            fusion.append(FusionD(config))

        self.fusion = nn.ModuleList(fusion)

        self.tcm = TCM(channels=config.channels)

        predict = []
        for i in range(4):
            predict.append(nn.Conv2d(config.channels, 1, 1))
        self.predict = nn.ModuleList(predict)

    def _init_weight(self, pretrained):
        pretrained_state_dict = torch.load(pretrained)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        state_dict = self.state_dict()
        all_params = {}
        for k, v in self.state_dict().items():
            if k in pretrained_state_dict.keys():
                v = pretrained_state_dict[k]
                all_params[k] = v

        state_dict.update(all_params)
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        x_size = x.shape[2:]
        x4, attn_weights, features = self.encoder(x)
        x3, x2, x1 = features

        x1 = self.compress[0](x1)
        x2 = self.compress[1](x2)
        x3 = self.compress[2](x3)
        x4 = self.compress[3](x4)

        x4 = self.tcm(x4)

        x3 = self.fusion[0](x3, x4)
        x2 = self.fusion[1](x2, x3, x4)
        x1 = self.fusion[2](x1, x2, x4)

        x1 = self.predict[0](x1)
        x2 = self.predict[1](x2)
        x3 = self.predict[2](x3)
        x4 = self.predict[3](x4)

        x1_pred = F.interpolate(x1, x_size, mode='bilinear', align_corners=True)
        x2_pred = F.interpolate(x2, x_size, mode='bilinear', align_corners=True)
        x3_pred = F.interpolate(x3, x_size, mode='bilinear', align_corners=True)
        x4_pred = F.interpolate(x4, x_size, mode='bilinear', align_corners=True)
        return torch.sigmoid(x1_pred), torch.sigmoid(x2_pred), torch.sigmoid(x3_pred), torch.sigmoid(x4_pred)
