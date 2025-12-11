import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- time embedding ----
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] scalar timesteps in [0, K-1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class TimeMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


# ---- blocks ----
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=32, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = in_ch != out_ch
        if self.skip:
            self.conv_skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        if self.skip:
            x = self.conv_skip(x)
        return x + h


class Down(nn.Module):
    def __init__(self, ch_in, ch_out, time_dim, num_res=2):
        super().__init__()
        blocks = []
        ch = ch_in
        for _ in range(num_res):
            blocks.append(ResBlock(ch, ch_out, time_dim))
            ch = ch_out
        self.block = nn.ModuleList(blocks)
        self.down = nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1)

    def forward(self, x, t):
        skips = []
        for b in self.block:
            x = b(x, t)
            skips.append(x)
        x = self.down(x)
        return x, skips


class Up(nn.Module):
    # curr_ch = channels of x entering this Up stage
    # skip_ch = channels of the skip tensors at this spatial size
    # out_ch  = channels you want after each block in this stage
    def __init__(self, curr_ch, skip_ch, out_ch, time_dim, num_res=2, do_upsample=True, debug=False):
        super().__init__()
        self.debug = debug
        # one fuse per block: first needs ch_in_after_concat, the rest need out_ch+out_ch
        self.fuse = nn.ModuleList([nn.Conv2d(curr_ch + skip_ch, out_ch, 1)] + [nn.Conv2d(out_ch + skip_ch, out_ch, 1) for _ in range(num_res - 1)])
        self.blocks = nn.ModuleList([ResBlock(out_ch, out_ch, time_dim) for _ in range(num_res)])
        self.up = (
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )
            if do_upsample
            else nn.Identity()
        )

    def forward(self, x, skip_list, t):
        for fuse, block in zip(self.fuse, self.blocks):

            s = skip_list.pop()

            if self.debug:
                print("[UP] input x:", x.shape)
                print("[UP] input skip:", s.shape)

            # make spatial sizes match (e.g., 4x4 -> 8x8 before concat)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")

            if self.debug:
                print("[UP] (maybe) interpolated x, skip:", x.shape, s.shape)

            x = torch.cat([x, s], dim=1)  # (curr or out_ch) + skip_ch
            x = fuse(x)  # → out_ch
            x = block(x, t)  # norm now sees the right channel count
        x = self.up(x)
        return x


# ---- unet ----
class UNetCifar32(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, channel_mults=(1, 2, 2, 2), num_res_blocks=2, dropout=0.1, time_hidden=512, gn_groups=32, debug=False):
        super().__init__()

        self.debug = debug

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_hidden),
            TimeMLP(time_hidden, time_hidden),
        )

        chs = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        in_ch = chs[0]
        for out_ch in chs[1:]:
            self.downs.append(Down(in_ch, out_ch, time_hidden, num_res_blocks))
            in_ch = out_ch

        self.mid = ResBlock(in_ch, in_ch, time_hidden, groups=gn_groups, dropout=dropout)

        # chs like [32, 64, 64, 64] for your run
        rev_pairs = list(reversed(list(zip(chs[:-1], chs[1:]))))  # (out_ch_at_stage, skip_ch_at_stage)
        curr = chs[-1]  # channels coming out of mid = 64

        self.ups = nn.ModuleList()
        for i, (out_ch, skip_ch) in enumerate(rev_pairs):
            do_up = i < len(rev_pairs) - 1  # last stage: no upsample
            if self.debug:
                print("[Build UP]: do_up flag", do_up)
            self.ups.append(Up(curr, skip_ch, out_ch, time_hidden, num_res_blocks, do_upsample=do_up))
            curr = out_ch

        self.out_norm = nn.GroupNorm(gn_groups, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(chs[0], out_channels, 3, padding=1)

    def forward(self, x, t):  # t can be [B] or [B,1]
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = self.time_emb(t)

        x = self.in_conv(x)
        skips_all = []
        h = x

        if self.debug:
            print("init:", x.shape)
            print("skips_all:", skips_all)
            print("h:", h.shape)

        for down in self.downs:
            h, skips = down(h, t_emb)
            skips_all.extend(skips)
            if self.debug:
                print("h [down]:", h.shape)
        h = self.mid(h, t_emb)
        if self.debug:
            print("h [mid]:", h.shape)
        for up in self.ups:
            h = up(h, skips_all, t_emb)
            if self.debug:
                print("h [up]:", h.shape)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        if self.debug:
            print("h [out]:", h.shape)

        return h


def build_unet_model(cfg, **kw):  # cfg is your full experiment cfg
    model_cfg = cfg.model          
    name = model_cfg.name

    if name == "unet_cifar32":
        return UNetCifar32(
            in_channels=kw.get("in_channels", model_cfg.in_channels),
            out_channels=kw.get("out_channels", model_cfg.out_channels),
            base_channels=kw.get("base_channels", model_cfg.base_channels),
            channel_mults=tuple(kw.get("channel_mults", model_cfg.channel_mults)),
            num_res_blocks=kw.get("num_res_blocks", model_cfg.num_res_blocks),
            dropout=kw.get("dropout", model_cfg.dropout),
            time_hidden=kw.get("time_embedding", model_cfg.time_embedding),
            gn_groups=kw.get("gn_groups", model_cfg.gn_groups),
        )

    raise KeyError(name)
