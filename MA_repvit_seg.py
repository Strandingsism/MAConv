import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from MAConv2D import MAConv2D

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class MAConv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.add_module('c', MAConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            manifold_groups=4, 
            freq_bands=6, 
            bias=False  
        ))
        self.add_module('bn', nn.BatchNorm2d(out_channels))

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

class MARepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(MARepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                MAConv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                MAConv2d_BN(inp, oup, 1, stride=1, padding=0)
            )
            self.channel_mixer = nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            )
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            )

    def forward(self, x):
        if self.identity:
            return x + self.channel_mixer(self.token_mixer(x))
        else:
            x1 = self.token_mixer(x)
            x2 = self.channel_mixer(x1)
            return x2

class PPM(nn.Module):
    def __init__(self, in_channels, pool_scales, norm_layer, out_channels=256):
        super(PPM, self).__init__()
        self.paths = nn.ModuleList()
        for scale in pool_scales:
            self.paths.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                Conv2d_BN(in_channels, out_channels, 1)
            ))

    def forward(self, x):
        input_size = x.size()
        out = []
        for path in self.paths:
            out.append(F.interpolate(
                path(x), size=input_size[-2:], mode='bilinear', align_corners=True))
        return out
        
class UPerHead(nn.Module):
    def __init__(self, in_channels, fpn_channels=256, out_channels=512):
        super(UPerHead, self).__init__()
        self.ppm = PPM(
            in_channels[-1],  
            [1, 2, 3, 6],
            norm_layer=nn.BatchNorm2d,
            out_channels=fpn_channels  
        )
        
        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        
        for i, ch in enumerate(in_channels): 
            self.fpn_in.append(Conv2d_BN(ch, fpn_channels, 1))
            self.fpn_out.append(Conv2d_BN(fpn_channels, fpn_channels, 3, 1, 1))
        self.fpn_bottleneck = Conv2d_BN(4 * fpn_channels, out_channels, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        """
        Args:
            features: list of features with channels [64, 128, 256, 512]
        """

        deepest_feat = features[-1]
        ppm_outs = [self.fpn_in[-1](deepest_feat)]  
        
        # 处理PPM输出
        for path in self.ppm.paths:
            ppm_out = path(deepest_feat)
            ppm_out = F.interpolate(
                ppm_out, 
                size=deepest_feat.shape[-2:],
                mode='bilinear', 
                align_corners=True
            )
            ppm_outs.append(ppm_out)

        f = self.fpn_out[-1](sum(ppm_outs)) 

        fpn_features = [f]
        
        for i in reversed(range(len(features) - 1)):
            lateral = self.fpn_in[i](features[i]) 
            f = lateral + F.interpolate(
                f, 
                size=lateral.shape[-2:],
                mode='bilinear', 
                align_corners=True
            )
            fpn_features.append(self.fpn_out[i](f))
 
        fpn_features.reverse()

        output_size = fpn_features[0].shape[-2:]
        out = [fpn_features[0]]
        
        for feat in fpn_features[1:]:
            out.append(F.interpolate(
                feat,
                size=output_size,
                mode='bilinear',
                align_corners=True
            ))
            
        out = torch.cat(out, dim=1)
        out = self.fpn_bottleneck(out)  
        out = self.dropout(out)
        return out

class MARepViTSeg(nn.Module):
    def __init__(self, cfgs, num_classes=19):
        super(MARepViTSeg, self).__init__()

        input_channel = cfgs[0][2]
        self.patch_embed = nn.Sequential(
            MAConv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            MAConv2d_BN(input_channel // 2, input_channel, 3, 2, 1)
        )
 
        self.stages = nn.ModuleList()
        block = MARepViTBlock

        self.downsample_indices = []  
        self.stage_channels = []      
        
        for stage_idx, (k, t, c, use_se, use_hs, s) in enumerate(cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            stage = block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            self.stages.append(stage)

            if s == 2 and stage_idx > 0:
                self.downsample_indices.append(stage_idx)
                self.stage_channels.append(output_channel)
                print(f"Downsample Stage {stage_idx}: channels={output_channel}")
            
            input_channel = output_channel  
        
        self.stage_channels.append(input_channel)
        print(f"Final Stage Channels: {self.stage_channels}")
        
        if len(self.stage_channels) < 4:
            self.stage_channels += [self.stage_channels[-1]] * (4 - len(self.stage_channels))
        else:
            self.stage_channels = self.stage_channels[-4:]  
        
        print(f"UPerHead Input Channels: {self.stage_channels}")
        
        self.decode_head = UPerHead(
            in_channels=self.stage_channels,
            fpn_channels=256,
            out_channels=512
        )
        
        self.seg_head = nn.Sequential(
            Conv2d_BN(512, 512, 3, 1, 1),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Backbone forward
        x = self.patch_embed(x)
        
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        decoder_indices = self.downsample_indices[-len(self.stage_channels):] 
        decoder_features = [features[i] for i in decoder_indices]

        while len(decoder_features) < 4:
            decoder_features.append(features[-1])
        
        # Decode head
        x = self.decode_head(decoder_features)
        
        # Segmentation head
        x = self.seg_head(x)
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        return x

def ma_repvit_seg_m0_9(num_classes=19, pretrained=False):
    """
    MA-RepViT-M0.9 for Cityscapes segmentation
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],  # P2
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],  # P3
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 384, 0, 1, 2],  # P4
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1]
    ]
    model = MARepViTSeg(cfgs, num_classes=num_classes)
    return model

def ma_repvit_seg_m1_0(num_classes=19, pretrained=False):
    """
    MA-RepViT-M1.0 for Cityscapes segmentation
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 2, 56, 1, 0, 1],
        [3, 2, 56, 0, 0, 1],
        [3, 2, 56, 0, 0, 1],
        [3, 2, 112, 0, 0, 2],  # P2
        [3, 2, 112, 1, 0, 1],
        [3, 2, 112, 0, 0, 1],
        [3, 2, 112, 0, 0, 1],
        [3, 2, 224, 0, 1, 2],  # P3
        [3, 2, 224, 1, 1, 1],
        [3, 2, 224, 0, 1, 1],
        [3, 2, 224, 1, 1, 1],
        [3, 2, 448, 0, 1, 2],  # P4
        [3, 2, 448, 1, 1, 1],
        [3, 2, 448, 0, 1, 1]
    ]
    model = MARepViTSeg(cfgs, num_classes=num_classes)
    return model 

def ma_repvit_seg_m1_5(num_classes=19, pretrained=False):
    """
    MA-RepViT-M1.5 for Cityscapes segmentation
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],  # P2
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],  # P3
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 512, 0, 1, 2],  # P4
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1]
    ]
    model = MARepViTSeg(cfgs, num_classes=num_classes)
    return model 