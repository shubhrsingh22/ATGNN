"""ATGNN: Audio Tagging Graph Neural Network (https://arxiv.org/abs/2311.01526).

The PGN (Patch GNN) encoder is the Pyramid Vision GNN of Han et al. (NeurIPS 2022),
operating on log-mel spectrogram patches. The optional MLG blocks add:
  - PLG (Patch-Label GNN): learnable label embeddings attend to their k nearest
    patch nodes via max-relative graph convolution.
  - LLG (Label-Label GNN): label embeddings are refined with a learnable
    adjacency matrix, L_hat = A @ L + L.
The final score combines patch logits (GAP + conv head) with a per-class readout
of the label embeddings.

With `use_mlg=False` the model reduces to the encoder-only variant
("ATGNN-pyr-s (-MLG)" in the paper).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from src.models.gcn_lib.torch_nn import BasicConv, act_layer, batched_index_select
from src.models.gcn_lib.torch_edge import xy_dense_knn_matrix
from src.models.gcn_lib.torch_vertex import Grapher


class Stem(nn.Module):
    """Two stride-2 convolutions reducing the spectrogram by 4x. Key names match
    the Pyramid ViG ImageNet checkpoint (stem.convs.*)."""

    def __init__(self, in_dim=1, out_dim=80, act='gelu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


class Downsample(nn.Module):
    """Stride-2 conv downsampling. Named `conv` to match the checkpoint keys
    (backbone.N.conv.*), so ImageNet weights load into it."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(x)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class PLGConv(nn.Module):
    """Patch-Label max-relative graph convolution: each label node aggregates
    from its k nearest patch nodes (paper eqs. 4-5)."""

    def __init__(self, channels, k=9, act='gelu', norm='batch', bias=True):
        super().__init__()
        self.k = k
        self.nn = BasicConv([channels * 2, channels], act, norm, bias)

    def forward(self, lab_x, patch_x):
        # lab_x: (B, C, S, 1), patch_x: (B, C, N, 1)
        lab_n = F.normalize(lab_x, p=2.0, dim=1)
        patch_n = F.normalize(patch_x, p=2.0, dim=1)
        edge_index = xy_dense_knn_matrix(lab_n, patch_n, self.k)
        x_i = batched_index_select(lab_x, edge_index[1])
        x_j = batched_index_select(patch_x, edge_index[0])
        x_u, _ = torch.max(x_i - x_j, -1, keepdim=True)
        b, c, s, _ = lab_x.shape
        out = torch.cat([lab_x.unsqueeze(2), x_u.unsqueeze(2)], dim=2).reshape(b, 2 * c, s, 1)
        return self.nn(out)


class MLGBlock(nn.Module):
    """One MLG block: PLG (patch-label) + FFN, followed by LLG (label-label)."""

    def __init__(self, channels, num_classes, k=9, act='gelu', norm='batch', bias=True, drop_path=0.0):
        super().__init__()
        self.plg = PLGConv(channels, k=k, act=act, norm=norm, bias=bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = FFN(channels, channels * 4, act=act, drop_path=drop_path)
        # LLG: learnable adjacency over label nodes
        self.adj = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.uniform_(self.adj, -1.0 / math.sqrt(num_classes), 1.0 / math.sqrt(num_classes))

    def forward(self, lab_x, patch_x):
        # PLG with residual
        lab_x = self.drop_path(self.plg(lab_x, patch_x)) + lab_x
        lab_x = self.ffn(lab_x)
        # LLG: L_hat = A @ L + L  (lab_x: B, C, S, 1)
        lab = lab_x.squeeze(-1).transpose(1, 2)  # (B, S, C)
        lab = torch.matmul(self.adj, lab) + lab
        return lab.transpose(1, 2).unsqueeze(-1)


class ATGNN(nn.Module):
    """Pyramid ATGNN. `use_mlg=False` gives the PGN-encoder-only variant."""

    def __init__(self, k=9, k_plg=9, act='gelu', norm='batch', bias=True, dropout=0.0,
                 epsilon=0.2, use_stochastic=False, drop_path=0.1, size='s',
                 num_class=200, freq_num=128, time_num=1024, conv='mr',
                 use_mlg=True, mlg_blocks=None, imagenet_shape=False):
        super().__init__()

        if size == 't':
            blocks, channels = [2, 2, 6, 2], [48, 96, 240, 384]
        elif size == 's':
            blocks, channels = [2, 2, 6, 2], [80, 160, 400, 640]
        elif size == 'm':
            blocks, channels = [2, 2, 16, 2], [96, 192, 384, 768]
        else:
            blocks, channels = [2, 2, 18, 2], [128, 256, 512, 1024]

        if mlg_blocks is None:
            mlg_blocks = [1, 1, 3, 1] if size in ('t', 's') else [1, 1, 6, 1]

        self.blocks = blocks
        self.channels = channels
        self.num_class = num_class
        self.use_mlg = use_mlg
        self.n_blocks = sum(blocks)

        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(in_dim=1, out_dim=channels[0], act=act)
        HW = (freq_num // 4) * (time_num // 4)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], freq_num // 4, time_num // 4))

        # PGN backbone: flat Sequential matching the Pyramid ViG checkpoint layout
        # (Seq(Grapher, FFN) entries with Downsample modules between stages).
        self.backbone = nn.ModuleList([])
        # index of the backbone entry that ends each stage (to tap features for MLG)
        self.stage_ends = []
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, use_stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(in_features=channels[i], hidden_features=channels[i] * 4, act=act, drop_path=dpr[idx]),
                    )
                ]
                idx += 1
            self.stage_ends.append(len(self.backbone) - 1)
        self.backbone = nn.Sequential(*self.backbone)

        # patch prediction head (matches checkpoint keys prediction.*)
        self.prediction = nn.Sequential(
            nn.Conv2d(channels[-1], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Conv2d(1024, num_class, 1, bias=True),
        )

        if use_mlg:
            # learnable label embeddings entering stage 1
            self.label_embed = nn.Parameter(torch.zeros(1, channels[0], num_class, 1))
            nn.init.trunc_normal_(self.label_embed, std=0.02)
            # per-stage: projection of label embeddings to the stage width + MLG blocks
            self.label_proj = nn.ModuleList()
            self.mlg_stages = nn.ModuleList()
            for i in range(len(blocks)):
                self.label_proj.append(
                    nn.Identity() if i == 0 else nn.Sequential(
                        nn.Conv2d(channels[i - 1], channels[i], 1, bias=True),
                        nn.BatchNorm2d(channels[i]),
                    )
                )
                self.mlg_stages.append(nn.ModuleList([
                    MLGBlock(channels[i], num_class, k=k_plg, act=act, norm=norm, bias=bias, drop_path=drop_path)
                    for _ in range(mlg_blocks[i])
                ]))
            # per-class readout: y_i = w_i . l_i
            self.label_readout = nn.Parameter(torch.empty(num_class, channels[-1]))
            nn.init.uniform_(self.label_readout, -1.0 / math.sqrt(channels[-1]), 1.0 / math.sqrt(channels[-1]))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        # x: (B, T, F) log-mel
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)  # (B, 1, F, T)

        x = self.stem(x) + self.pos_embed

        if not self.use_mlg:
            for blk in self.backbone:
                x = blk(x)
            out = F.adaptive_avg_pool2d(x, 1)
            return self.prediction(out).squeeze(-1).squeeze(-1)

        lab = self.label_embed.expand(x.shape[0], -1, -1, -1)
        stage = 0
        for i, blk in enumerate(self.backbone):
            x = blk(x)
            if i == self.stage_ends[stage]:
                B, C, H, W = x.shape
                patch_nodes = x.reshape(B, C, -1, 1)
                lab = self.label_proj[stage](lab)
                for mlg in self.mlg_stages[stage]:
                    lab = mlg(lab, patch_nodes)
                stage += 1

        patch_logits = self.prediction(F.adaptive_avg_pool2d(x, 1)).squeeze(-1).squeeze(-1)
        # per-class readout of the refined label embeddings
        label_logits = torch.einsum('bcs,sc->bs', lab.squeeze(-1), self.label_readout)
        return patch_logits + label_logits
