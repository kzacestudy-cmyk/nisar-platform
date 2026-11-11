"""Inference, test-time augmentation, post-processing and polygon extraction."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, binary_opening
from skimage import measure

from .coherence import CoherenceAnalyzer
from .model import UNet, get_device


class DisasterDetector:
    def __init__(self, checkpoint_path: str | None = None, model: UNet | None = None,
                 device: torch.device | None = None):
        self.device = device or get_device()
        self.model = (model or UNet()).to(self.device)
        self.is_trained = model is not None
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"])
            self.is_trained = True
        self.model.eval()

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _stack(coherence_before, coherence_after, amplitude):
        x = np.stack([coherence_before, coherence_after, amplitude]).astype(np.float32)
        return torch.from_numpy(x)[None]

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to a multiple of 16, run the network, crop back."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = self.model(x.to(self.device)).cpu()
        return y[..., :h, :w]

    # -- public API -------------------------------------------------------
    def predict(self, coherence_before, coherence_after, amplitude) -> np.ndarray:
        """Return per-pixel probability map (H, W) in [0, 1]."""
        x = self._stack(coherence_before, coherence_after, amplitude)
        return self._forward(x)[0, 0].numpy()

    def ensemble_predict(self, coherence_before, coherence_after, amplitude) -> np.ndarray:
        """Test-time augmentation over 4 rotations, averaged."""
        x = self._stack(coherence_before, coherence_after, amplitude)
        preds = []
        for k in range(4):
            y = self._forward(torch.rot90(x, k, (2, 3)))
            preds.append(torch.rot90(y, -k, (2, 3)))
        return torch.stack(preds).mean(0)[0, 0].numpy()

    @staticmethod
    def postprocess(raw_mask, min_area_px: int = 50, threshold: float = 0.5) -> np.ndarray:
        """Threshold, open, fill holes and drop small blobs."""
        mask = np.asarray(raw_mask) > threshold if raw_mask.dtype != bool else raw_mask
        mask = binary_opening(mask, iterations=1)
        mask = binary_fill_holes(mask)
        labels = measure.label(mask, connectivity=2)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_area_px
        keep[0] = False
        return keep[labels]

    @staticmethod
    def pixel_to_lonlat(row, col, shape, lat_lon_bounds):
        """bounds = (lat_min, lat_max, lon_min, lon_max); row 0 is north."""
        lat_min, lat_max, lon_min, lon_max = lat_lon_bounds
        h, w = shape
        lat = lat_max - (row / (h - 1)) * (lat_max - lat_min)
        lon = lon_min + (col / (w - 1)) * (lon_max - lon_min)
        return lon, lat

    @staticmethod
    def pixel_area_km2(shape, lat_lon_bounds):
        lat_min, lat_max, lon_min, lon_max = lat_lon_bounds
        h, w = shape
        lat_c = np.deg2rad((lat_min + lat_max) / 2)
        dy = (lat_max - lat_min) * 111.32 / h
        dx = (lon_max - lon_min) * 111.32 * np.cos(lat_c) / w
        return dx * dy

    def extract_event_polygons(self, mask, lat_lon_bounds=None, displacement_mm=None,
                               delta_coh=None, simplify_tol: float = 1.5) -> dict:
        """Convert a binary mask into a GeoJSON FeatureCollection.

        Coordinates are lon/lat if `lat_lon_bounds` is given, otherwise pixels.
        Each feature carries area, max subsidence, coherence loss and risk.
        """
        mask = np.asarray(mask, dtype=bool)
        labels = measure.label(mask, connectivity=2)
        shape = mask.shape
        px_km2 = self.pixel_area_km2(shape, lat_lon_bounds) if lat_lon_bounds else None

        global_disp_max = float(np.abs(displacement_mm).max()) if displacement_mm is not None else 1.0
        global_dc_max = float(np.clip(delta_coh, 0, None).max()) if delta_coh is not None else 1.0

        features = []
        for i, region in enumerate(measure.regionprops(labels), start=1):
            region_mask = np.pad(labels == region.label, 1)
            contours = measure.find_contours(region_mask.astype(float), 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            contour = measure.approximate_polygon(contour, tolerance=simplify_tol) - 1  # undo pad
            if lat_lon_bounds:
                ring = [list(self.pixel_to_lonlat(r, c, shape, lat_lon_bounds)) for r, c in contour]
            else:
                ring = [[float(c), float(r)] for r, c in contour]
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            cy, cx = region.centroid
            props = {
                "event_id": f"EV-{i:03d}",
                "area_px": int(region.area),
                "centroid_row": float(cy),
                "centroid_col": float(cx),
            }
            if lat_lon_bounds:
                lon, lat = self.pixel_to_lonlat(cy, cx, shape, lat_lon_bounds)
                props.update(lat=round(float(lat), 5), lon=round(float(lon), 5),
                             area_km2=round(float(region.area * px_km2), 3))
            sel = labels == region.label
            disp_score = dc_score = 0.0
            if displacement_mm is not None:
                vals = displacement_mm[sel]
                props["max_subsidence_mm"] = round(float(vals.min()), 1)
                props["mean_displacement_mm"] = round(float(vals.mean()), 1)
                disp_score = float(np.abs(vals).max()) / max(global_disp_max, 1e-9)
            if delta_coh is not None:
                dc = delta_coh[sel]
                props["mean_delta_coh"] = round(float(dc.mean()), 3)
                dc_score = float(np.clip(dc, 0, None).max()) / max(global_dc_max, 1e-9)
            score = disp_score * 60 + dc_score * 40
            level = CoherenceAnalyzer.risk_level(score)
            props.update(risk_score=round(float(score), 1), risk_level=level)

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            })

        features.sort(key=lambda f: -f["properties"]["risk_score"])
        return {"type": "FeatureCollection", "features": features}
