import torch
import torch.nn as nn
import torch.nn.functional as F
#my model
class RGBPhysicalInput(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, rgb):
        rgb = rgb.clamp(min=self.eps)
        R = rgb[:, 0:1]
        G = rgb[:, 1:2]
        B = rgb[:, 2:3]
        rgb_sum = R + G + B + self.eps
        chroma = rgb / rgb_sum
        log_rg = torch.log(R / (G + self.eps))
        log_bg = torch.log(B / (G + self.eps))
        intensity = rgb.mean(dim=1, keepdim=True)
        return torch.cat([rgb, chroma, log_rg, log_bg, intensity], dim=1)

class SCRU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, num_res=1, dilation=1):
        super().__init__()
        self.num_res = num_res
        num_groups = 8 if out_ch % 8 == 0 else 4
        self.proj_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.proj_gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.proj_act = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList()
        for _ in range(num_res):
            block = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            )
            self.res_blocks.append(block)

        self.res_act = nn.ReLU(inplace=True)

    def forward(self, x):
        #
        x = self.proj_conv(x)
        x = self.proj_gn(x)
        x = self.proj_act(x)
        for block in self.res_blocks:
            identity = x
            x = block(x)
            x = self.res_act(x + identity)
        return x

class RGBSpatialEncoder(nn.Module):
    def __init__(self, feat_dim=512, use_physical_input=True):
        super().__init__()
        self.use_physical_input = use_physical_input
        # Physical RGB input features.
        if use_physical_input:
            self.physical_input = RGBPhysicalInput()
            in_channels = 9
        else:
            self.physical_input = nn.Identity()
            in_channels = 3
        # ========== Four stages built from standard SCRU units ==========
        # 9 -> 32 channels.
        self.scru0 = SCRU(in_channels, 32, stride=2, num_res=0)
        # 32 -> 64 channels.
        self.scru1 = SCRU(32, 64, stride=2, num_res=0)
        # 64→128
        self.scru2 = SCRU(64, 128, stride=2, num_res=1, dilation=1)
        # 128→256
        self.scru3 = SCRU(128, 256, stride=1, num_res=2, dilation=1)
        #
        num_groups = 8 if feat_dim % 8 == 0 else 4
        self.out_proj = nn.Sequential(
            nn.Conv2d(256, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, rgb):
        x = self.physical_input(rgb)

        x = self.scru0(x)  # 32 channels, 1/2 spatial resolution.
        x = self.scru1(x)  # 64 channels, 1/4 spatial resolution.
        x = self.scru2(x)  # 128 channels, 1/8 spatial resolution.
        x = self.scru3(x)  # 256 channels, 1/8 spatial resolution.

        fmap = self.out_proj(x)
        feat = self.pool(fmap).flatten(1)
        return feat

# Multispectral branch.
class ResidualDWBlock(nn.Module):
    def __init__(self, channels, dilation=1, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = activation(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return self.act(out + identity)

class GlobalContextBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        g = self.pool(x)
        g = self.proj(g)
        return g.expand(-1, -1, x.shape[2], x.shape[3])


class BranchWeightPredictor(nn.Module):
    def __init__(self, in_channels, num_branches=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_branches)
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=1)

class Spectral1DEnhance(nn.Module):
    def __init__(self, bands=16, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous()
        seq = seq.view(b * h * w, 1, c)
        out = self.net(seq)
        out = out.view(b, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()
        return x + self.scale * out

class MSMultiBranchStableBackbone(nn.Module):
    def __init__(self, in_channels=16, feat_dim=512, branch_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.spectral_enhance = Spectral1DEnhance(
            bands=in_channels,
            hidden=16,
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.ReLU(inplace=True),
            ResidualDWBlock(branch_dim, dilation=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.ReLU(inplace=True),
            ResidualDWBlock(branch_dim, dilation=2),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.ReLU(inplace=True),
            ResidualDWBlock(branch_dim, dilation=3),
        )
        self.branch4 = nn.Sequential(
            GlobalContextBranch(in_channels, branch_dim),
            ResidualDWBlock(branch_dim, dilation=1),
        )
        self.weight_pred = BranchWeightPredictor(
            in_channels,
            num_branches=4,
            hidden=64,
        )
        self.post_fuse = nn.Sequential(
            nn.Conv2d(branch_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_weights=False, return_map=False):
        assert x.dim() == 4, f"MSI should be [B, C, H, W], but got {x.shape}"
        assert x.shape[1] == self.in_channels, (
            f"Expected MSI channels={self.in_channels}, but got {x.shape[1]}"
        )
        x = self.spectral_enhance(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        branches = torch.stack([b1, b2, b3, b4], dim=1)
        w = self.weight_pred(x)
        w_expand = w[:, :, None, None, None]
        fused = torch.sum(branches * w_expand, dim=1)
        fused = self.post_fuse(fused)
        if return_map:
            return fused
        feat = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        if return_weights:
            return feat, w
        return feat

class ResidualFusionBlock(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, msi_feat):
        rgb_corr = self.rgb_proj(rgb_feat)
        alpha = self.gate(torch.cat([rgb_feat, msi_feat], dim=1))
        fused_feat = msi_feat + alpha * rgb_corr
        return fused_feat, alpha

#
class FusionSPDNet(nn.Module):
    def __init__(
            self,
            spd_dim=36,
            msi_bands=16,
            feat_dim=512,
            use_physical_rgb=True,
            positive_output=False,
            normalize_spd=None,
    ):
        super().__init__()
        self.positive_output = positive_output
        self.normalize_spd = normalize_spd
        self.rgb_encoder = RGBSpatialEncoder(
            feat_dim=feat_dim,
            use_physical_input=use_physical_rgb
        )
        self.msi_encoder = MSMultiBranchStableBackbone(
            in_channels=msi_bands,
            feat_dim=feat_dim,
            branch_dim=128
        )
        self.rgb_norm = nn.LayerNorm(feat_dim)
        self.msi_norm = nn.LayerNorm(feat_dim)
        self.fusion_block = ResidualFusionBlock(
            feat_dim=feat_dim
        )
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, spd_dim)
        )

    def forward(self, rgb, msi):
        rgb_feat = self.rgb_encoder(rgb)
        msi_feat = self.msi_encoder(msi)
        rgb_feat = self.rgb_norm(rgb_feat)
        msi_feat = self.msi_norm(msi_feat)
        fused_feat, alpha = self.fusion_block(rgb_feat, msi_feat)
        spd = self.regressor(fused_feat)
        if self.positive_output:
            spd = F.softplus(spd)
        if self.normalize_spd == "sum":
            spd = spd / (spd.sum(dim=1, keepdim=True) + 1e-6)
        elif self.normalize_spd == "max":
            spd = spd / (spd.max(dim=1, keepdim=True).values + 1e-6)
        return spd

# ========== Simple forward-pass example ==========
if __name__ == "__main__":
    model = FusionSPDNet()
    ms = torch.randn(2, 16, 256, 256)
    rgb = torch.randn(2, 3, 256, 256)
    pred_spectrum = model(rgb, ms)

    print(f"预测光谱形状: {pred_spectrum.shape}")  # [2, 36]
    print("模型前向传播正常")