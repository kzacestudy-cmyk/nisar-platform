"""U-Net for disaster-zone segmentation from coherence stacks."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_opening, binary_fill_holes


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Conv block, then MaxPool. Returns (skip, pooled)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """ConvTranspose2d upsample, concat skip, conv block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """4-level U-Net. Input: 3 channels (coh_before, coh_after, amplitude)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = EncoderBlock(in_channels, c[0])
        self.enc2 = EncoderBlock(c[0], c[1])
        self.enc3 = EncoderBlock(c[1], c[2])
        self.enc4 = EncoderBlock(c[2], c[3])
        self.bottleneck = ConvBlock(c[3], c[3] * 2)
        self.dec4 = DecoderBlock(c[3] * 2, c[3])
        self.dec3 = DecoderBlock(c[3], c[2])
        self.dec2 = DecoderBlock(c[2], c[1])
        self.dec1 = DecoderBlock(c[1], c[0])
        self.head = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, x, return_logits: bool = False):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        logits = self.head(x)
        return logits if return_logits else torch.sigmoid(logits)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ModelTrainer:
    def __init__(self, model: UNet | None = None, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = (model or UNet()).to(self.device)
        self.tiles_x: np.ndarray | None = None
        self.tiles_y: np.ndarray | None = None
        self.history: list[float] = []

    # -- data -------------------------------------------------------------
    @staticmethod
    def _tile_starts(length: int, tile: int, stride: int):
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile + 1, stride))
        if starts[-1] != length - tile:
            starts.append(length - tile)
        return starts

    def prepare_tiles(self, coherence_stack, displacement, tile_size: int = 256, overlap: int = 32,
                      labels=None):
        """Cut (C,H,W) input stack and labels into overlapping tiles.

        `labels` are pseudo-labels; if omitted they are derived from displacement
        and the coherence change of the stack (channel0 - channel1).
        """
        x = np.asarray(coherence_stack, dtype=np.float32)
        if labels is None:
            labels = self.create_pseudo_labels(displacement, x[0] - x[1])
        y = np.asarray(labels, dtype=np.float32)
        _, h, w = x.shape
        stride = tile_size - overlap
        xs, ys = [], []
        for r in self._tile_starts(h, tile_size, stride):
            for c in self._tile_starts(w, tile_size, stride):
                xs.append(x[:, r:r + tile_size, c:c + tile_size])
                ys.append(y[None, r:r + tile_size, c:c + tile_size])
        self.tiles_x = np.stack(xs)
        self.tiles_y = np.stack(ys)
        return self.tiles_x, self.tiles_y

    @staticmethod
    def create_pseudo_labels(displacement_mm, delta_coh, disp_frac: float = 0.25,
                             coh_thr: float = 0.2):
        """Weak labels: strong deformation AND/OR strong decorrelation."""
        disp = np.abs(np.asarray(displacement_mm))
        disp_thr = disp_frac * disp.max() if disp.max() > 0 else np.inf
        # Remove the large-scale background before thresholding displacement
        strong_def = disp > max(disp_thr, np.percentile(disp, 90))
        strong_dec = np.asarray(delta_coh) > coh_thr
        label = (strong_def & (np.asarray(delta_coh) > coh_thr * 0.5)) | strong_dec
        label = binary_opening(label, iterations=2)
        label = binary_fill_holes(label)
        return label.astype(np.float32)

    # -- training ---------------------------------------------------------
    def _augment(self, xb: torch.Tensor, yb: torch.Tensor):
        k = int(np.random.randint(4))
        xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (2, 3))
        if np.random.rand() < 0.5:
            xb, yb = torch.flip(xb, (3,)), torch.flip(yb, (3,))
        return xb, yb

    def train(self, epochs: int = 20, lr: float = 1e-4, batch_size: int = 4,
              streamlit_log: bool = True, progress_callback=None):
        """Train on prepared tiles with BCE + Dice loss.

        Loss is logged live to Streamlit (progress bar + line chart) when
        running inside a Streamlit app and `streamlit_log` is True.
        """
        if self.tiles_x is None:
            raise RuntimeError("Call prepare_tiles() first.")

        st_bar = st_chart = st_text = None
        if streamlit_log:
            try:
                import streamlit as st
                from streamlit.runtime.scriptrunner import get_script_run_ctx

                if get_script_run_ctx() is not None:
                    st_text = st.empty()
                    st_bar = st.progress(0.0)
                    st_chart = st.empty()
            except Exception:
                pass

        X = torch.from_numpy(self.tiles_x)
        Y = torch.from_numpy(self.tiles_y)
        bce = nn.BCEWithLogitsLoss()  # Dice term handles class imbalance
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        n = len(X)
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = self._augment(X[idx], Y[idx])
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb, return_logits=True)
                prob = torch.sigmoid(logits)
                inter = (prob * yb).sum()
                dice = 1 - (2 * inter + 1) / (prob.sum() + yb.sum() + 1)
                loss = bce(logits, yb) + dice
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            epoch_loss = total / n
            self.history.append(epoch_loss)

            if st_bar is not None:
                st_bar.progress(epoch / epochs)
                st_text.markdown(f"**Epoch {epoch}/{epochs}** — loss `{epoch_loss:.4f}`")
                st_chart.line_chart({"loss": self.history}, height=180)
            if progress_callback is not None:
                progress_callback(epoch, epochs, epoch_loss)

        self.model.eval()
        return self.history

    # -- checkpoints --------------------------------------------------------
    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "history": self.history}, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.history = ckpt.get("history", [])
        self.model.eval()
        return self.model
