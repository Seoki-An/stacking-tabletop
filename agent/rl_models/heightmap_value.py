"""
U-Net value model over height maps. Takes (scene, target) height maps and
emits a coarse score map interpreted as Q(s, xy) at a downsampled grid. Drops
in for `PlanarPoseSampler._score_map`.

Asymmetric U-Net: the encoder downsamples down to a bottleneck (deeper than
`output.resolution`), the decoder upsamples back up to `output.resolution`
with skip connections from the matching encoder stages. The output spatial
size therefore stays compatible with the heuristic baseline (and with
`PlanarPoseSampler._topk_xy` consumption), independent of how deep the
encoder runs.

Required: `n_encoder_stages - n_decoder_stages == log2(input_res / output_res)`.
With `decoder.channels: []` (or absent) the model collapses to the legacy
encoder-only behaviour for backwards-compatible configs.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class _DownStage(nn.Module):
    """`convs_per_stage` Conv-Act blocks, then 2× MaxPool. Returns
    `(pooled, pre_pool)` so the matching decoder stage can wire a skip."""

    def __init__(self, in_ch: int, out_ch: int, n_convs: int, kernel: int, act_cls):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(n_convs):
            c_in = in_ch if i == 0 else out_ch
            layers.extend(
                [
                    nn.Conv2d(c_in, out_ch, kernel, padding=kernel // 2),
                    act_cls(),
                ]
            )
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.convs(x)
        return self.pool(skip), skip


class _UpStage(nn.Module):
    """2× upsample, concat skip from the matching encoder stage, then
    `convs_per_stage` Conv-Act blocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        n_convs: int,
        kernel: int,
        act_cls,
        mode: str = "bilinear",
    ):
        super().__init__()
        if mode == "transposed":
            self.up: nn.Module = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2
            )
        elif mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        elif mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            raise ValueError(
                f"_UpStage: unknown upsample mode {mode!r} "
                f"(expected 'bilinear' | 'nearest' | 'transposed')"
            )

        layers: List[nn.Module] = []
        in_concat = in_ch + skip_ch
        for i in range(n_convs):
            c_in = in_concat if i == 0 else out_ch
            layers.extend(
                [
                    nn.Conv2d(c_in, out_ch, kernel, padding=kernel // 2),
                    act_cls(),
                ]
            )
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.convs(x)


class HeightmapValueModel(nn.Module):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cpu")

        in_ch = int(cfg.input.channels)
        out_ch = int(cfg.output.channels)
        h_in, w_in = int(cfg.input.resolution[0]), int(cfg.input.resolution[1])
        h_out, w_out = int(cfg.output.resolution[0]), int(cfg.output.resolution[1])

        if h_in != w_in or h_out != w_out:
            raise ValueError("input/output resolutions must be square")
        if h_in % h_out != 0:
            raise ValueError(
                f"input resolution {h_in} must be a multiple of output {h_out}"
            )
        ratio_log = int(np.log2(h_in // h_out))
        if 2**ratio_log * h_out != h_in:
            raise ValueError(
                "input/output resolution ratio must be a power of two"
            )

        # ----- encoder -----
        enc_channels = [int(c) for c in cfg.encoder.channels]
        n_enc = len(enc_channels)
        kernel = int(cfg.encoder.kernel_size)
        act_cls = _ACTIVATIONS[cfg.encoder.activation.lower()]
        enc_convs = int(cfg.encoder.get("convs_per_stage", 2))

        self.encoder_stages = nn.ModuleList()
        prev_ch = in_ch
        for stage_ch in enc_channels:
            self.encoder_stages.append(
                _DownStage(prev_ch, stage_ch, enc_convs, kernel, act_cls)
            )
            prev_ch = stage_ch

        # ----- bottleneck -----
        bottleneck_cfg = cfg.get("bottleneck", None)
        if bottleneck_cfg is not None:
            b_ch = int(bottleneck_cfg.channels)
            b_n = int(bottleneck_cfg.get("convs", 2))
            bottleneck_layers: List[nn.Module] = []
            for i in range(b_n):
                c_in = prev_ch if i == 0 else b_ch
                bottleneck_layers.extend(
                    [
                        nn.Conv2d(c_in, b_ch, kernel, padding=kernel // 2),
                        act_cls(),
                    ]
                )
            self.bottleneck = nn.Sequential(*bottleneck_layers)
            prev_ch = b_ch
        else:
            self.bottleneck = nn.Identity()

        # ----- decoder -----
        dec_cfg = cfg.get("decoder", None)
        dec_channels = (
            [int(c) for c in dec_cfg.channels]
            if dec_cfg is not None and "channels" in dec_cfg
            else []
        )
        n_dec = len(dec_channels)
        if n_enc - n_dec != ratio_log:
            raise ValueError(
                f"encoder/decoder mismatch: n_enc={n_enc}, n_dec={n_dec}, "
                f"but with input {h_in} and output {h_out} we need "
                f"n_enc - n_dec = log2({h_in}/{h_out}) = {ratio_log}. "
                f"Either trim encoder.channels or add decoder.channels."
            )

        if n_dec > 0:
            dec_convs = int(dec_cfg.get("convs_per_stage", enc_convs))
            dec_kernel = int(dec_cfg.get("kernel_size", kernel))
            dec_mode = str(dec_cfg.get("upsample", "bilinear"))
            self.decoder_stages = nn.ModuleList()
            # Decoder stage k (deepest → shallowest) consumes the encoder
            # pre-pool feature at index (n_enc - 1 - k). The deepest decoder
            # stage upsamples from the bottleneck (spatial size after the last
            # encoder pool) up to 2× that — which matches encoder stage
            # (n_enc - 1)'s pre-pool spatial size.
            for k, stage_ch in enumerate(dec_channels):
                skip_ch = enc_channels[n_enc - 1 - k]
                self.decoder_stages.append(
                    _UpStage(
                        in_ch=prev_ch,
                        skip_ch=skip_ch,
                        out_ch=stage_ch,
                        n_convs=dec_convs,
                        kernel=dec_kernel,
                        act_cls=act_cls,
                        mode=dec_mode,
                    )
                )
                prev_ch = stage_ch
        else:
            self.decoder_stages = nn.ModuleList()

        # ----- head -----
        head_layers: List[nn.Module] = []
        head_cfg = cfg.get("head", None)
        if head_cfg is not None:
            for h_ch in list(head_cfg.get("hidden", [])):
                head_layers.extend(
                    [
                        nn.Conv2d(prev_ch, int(h_ch), 1),
                        act_cls(),
                    ]
                )
                prev_ch = int(h_ch)
        head_layers.append(nn.Conv2d(prev_ch, out_ch, 1))
        self.head = nn.Sequential(*head_layers)

        n_params = sum(p.numel() for p in self.parameters())
        arch = "U-Net" if n_dec > 0 else "encoder-only"
        print(f"HeightmapValueModel ({arch}): {n_params/1e6:.2f}M params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H_in, W_in) -> (B, C_out, H_out, W_out)."""
        n_dec = len(self.decoder_stages)
        n_enc = len(self.encoder_stages)

        # Only keep the encoder pre-pool features that the decoder will use.
        skips: List[Optional[torch.Tensor]] = []
        for i, stage in enumerate(self.encoder_stages):
            x, skip = stage(x)
            skips.append(skip if i >= n_enc - n_dec else None)

        x = self.bottleneck(x)

        for k, dec_stage in enumerate(self.decoder_stages):
            skip = skips[n_enc - 1 - k]
            x = dec_stage(x, skip)

        return self.head(x)

    @torch.no_grad()
    def score_map(
        self, scene_height_map: np.ndarray, target_height_map: np.ndarray
    ) -> np.ndarray:
        """Inference helper for `PlanarPoseSampler`.

        Accepts two (H_in, W_in) height maps, returns a (H_out, W_out) score map
        as a numpy array (matching the contract of the legacy heuristic pipeline).
        """
        was_training = self.training
        self.eval()
        x = (
            torch.stack(
                [
                    torch.as_tensor(scene_height_map, dtype=torch.float32),
                    torch.as_tensor(target_height_map, dtype=torch.float32),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(self.device)
        )
        out = self.forward(x).squeeze(0).squeeze(0).cpu().numpy()
        if was_training:
            self.train()
        return out

    def set_device(self, device: str):
        self.device = torch.device(device)
        self.to(self.device)


class HeightmapFeasibilityModel(nn.Module):
    """Candidate feasibility model over height-map channels.

    Unlike `HeightmapValueModel`, this returns a scalar per candidate. The
    expected input is a local or global height-map stack such as
    `[scene, target, stone_bottom, stone_top, clearance]`.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cpu")

        in_ch = int(cfg.input.channels)
        enc_channels = [int(c) for c in cfg.encoder.channels]
        kernel = int(cfg.encoder.kernel_size)
        act_cls = _ACTIVATIONS[cfg.encoder.activation.lower()]
        enc_convs = int(cfg.encoder.get("convs_per_stage", 2))

        self.encoder_stages = nn.ModuleList()
        prev_ch = in_ch
        for stage_ch in enc_channels:
            self.encoder_stages.append(
                _DownStage(prev_ch, stage_ch, enc_convs, kernel, act_cls)
            )
            prev_ch = stage_ch

        bottleneck_cfg = cfg.get("bottleneck", None)
        if bottleneck_cfg is not None:
            b_ch = int(bottleneck_cfg.channels)
            b_n = int(bottleneck_cfg.get("convs", 2))
            layers: List[nn.Module] = []
            for i in range(b_n):
                c_in = prev_ch if i == 0 else b_ch
                layers.extend(
                    [
                        nn.Conv2d(c_in, b_ch, kernel, padding=kernel // 2),
                        act_cls(),
                    ]
                )
            self.bottleneck = nn.Sequential(*layers)
            prev_ch = b_ch
        else:
            self.bottleneck = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)

        point_cfg = cfg.get("point_encoder", None)
        point_hidden = (
            [int(value) for value in point_cfg.get("hidden", [])]
            if point_cfg is not None
            else []
        )
        point_layers: List[nn.Module] = []
        point_in = 6
        for point_out in point_hidden:
            point_layers.extend([nn.Linear(point_in, point_out), act_cls()])
            point_in = point_out
        self.point_encoder = nn.Sequential(*point_layers)
        self.point_coordinate_scale = float(
            point_cfg.get("coordinate_scale", 0.1)
            if point_cfg is not None
            else 0.1
        )
        if self.point_coordinate_scale <= 0.0:
            raise ValueError("point_encoder.coordinate_scale must be positive")
        self.point_feature_dim = 2 * point_in if point_hidden else 0
        prev_ch += 2 * self.point_feature_dim

        scalar_cfg = cfg.get("scalar_encoder", None)
        scalar_hidden = (
            [int(value) for value in scalar_cfg.get("hidden", [])]
            if scalar_cfg is not None
            else []
        )
        scalar_layers: List[nn.Module] = []
        scalar_in = 24
        if scalar_hidden:
            scalar_layers.append(nn.LayerNorm(scalar_in))
            for scalar_out in scalar_hidden:
                scalar_layers.extend([nn.Linear(scalar_in, scalar_out), act_cls()])
                scalar_in = scalar_out
        self.scalar_encoder = nn.Sequential(*scalar_layers)
        self.scalar_feature_dim = scalar_in if scalar_hidden else 0
        prev_ch += self.scalar_feature_dim

        head_layers: List[nn.Module] = []
        head_cfg = cfg.get("head", None)
        if head_cfg is not None:
            dropout = float(head_cfg.get("dropout", 0.0))
            for h_ch in list(head_cfg.get("hidden", [])):
                head_layers.extend([nn.Linear(prev_ch, int(h_ch)), act_cls()])
                if dropout > 0.0:
                    head_layers.append(nn.Dropout(p=dropout))
                prev_ch = int(h_ch)
        out_ch = int(cfg.output.get("channels", 1))
        head_layers.append(nn.Linear(prev_ch, out_ch))
        self.head = nn.Sequential(*head_layers)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"HeightmapFeasibilityModel: {n_params/1e6:.2f}M params")

    def forward(self, inputs) -> torch.Tensor:
        if isinstance(inputs, dict):
            heightmaps = inputs["heightmaps"]
        else:
            heightmaps = inputs
            inputs = {}

        x = heightmaps
        for stage in self.encoder_stages:
            x, _ = stage(x)
        x = self.bottleneck(x)
        features = [self.pool(x).flatten(1)]

        if self.point_feature_dim:
            features.append(self._encode_point_set(inputs, x, "candidate_dsf"))
            features.append(self._encode_point_set(inputs, x, "local_scene_dsf"))
        if self.scalar_feature_dim:
            features.append(self._encode_candidate_scalars(inputs, x))
        return self.head(torch.cat(features, dim=1))

    def _encode_point_set(
        self, inputs: dict, reference: torch.Tensor, prefix: str
    ):
        points = inputs.get(f"{prefix}_points")
        normals = inputs.get(f"{prefix}_normals")
        mask = inputs.get(f"{prefix}_point_mask")
        batch_size = reference.shape[0]
        if points is None or normals is None:
            return reference.new_zeros(batch_size, self.point_feature_dim)

        points = points.to(device=reference.device, dtype=reference.dtype)
        normals = normals.to(device=reference.device, dtype=reference.dtype)
        if mask is None:
            mask = torch.ones(
                points.shape[:2], dtype=torch.bool, device=reference.device
            )
        else:
            mask = mask.to(device=reference.device, dtype=torch.bool)
        mask_f = mask.unsqueeze(-1).to(reference.dtype)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        normalized_points = (points / self.point_coordinate_scale) * mask_f

        point_features = self.point_encoder(
            torch.cat([normalized_points, normals * mask_f], dim=-1)
        )
        masked = point_features.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = masked.amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        mean = (point_features * mask_f).sum(dim=1) / count
        return torch.cat([maximum, mean], dim=-1)

    def _encode_candidate_scalars(self, inputs: dict, reference: torch.Tensor):
        batch_size = reference.shape[0]

        def value(name: str, width: int):
            tensor = inputs.get(name)
            if tensor is None:
                return reference.new_zeros(batch_size, width)
            tensor = tensor.to(device=reference.device, dtype=reference.dtype)
            return tensor.reshape(batch_size, width)

        scalars = torch.cat(
            [
                value("candidate_physical_features", 14),
                value("candidate_pose", 7),
                value("c_feq", 1),
                value("c_gap", 1),
                value("depth", 1),
            ],
            dim=1,
        )
        return self.scalar_encoder(scalars)

    @torch.no_grad()
    def predict_proba(self, heightmaps: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        x = torch.as_tensor(heightmaps, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        logits = self.forward(x).squeeze(-1)
        out = torch.sigmoid(logits).cpu().numpy()
        if was_training:
            self.train()
        return out

    def set_device(self, device: str):
        self.device = torch.device(device)
        self.to(self.device)
