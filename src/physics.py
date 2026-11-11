"""SAR interferometry physics: phase -> displacement, synthetic GUNW generation."""

import numpy as np


class SARPhysics:
    """Physical conversions for L-band (NISAR) interferometric products."""

    def __init__(self, wavelength: float = 0.24, incidence_angle_deg: float = 35.0):
        self.wavelength = wavelength
        self.incidence_angle_deg = incidence_angle_deg

    @staticmethod
    def phase_to_displacement(phase_array, wavelength: float = 0.24):
        """Unwrapped phase (rad) -> line-of-sight displacement (mm)."""
        phase = np.asarray(phase_array, dtype=np.float64)
        return (phase * wavelength) / (4 * np.pi) * 1000.0

    @staticmethod
    def compute_los_to_vertical(los_mm, incidence_angle_deg: float = 35.0):
        """Project LOS displacement to vertical (assumes purely vertical motion)."""
        inc = np.deg2rad(incidence_angle_deg)
        return np.asarray(los_mm, dtype=np.float64) / np.cos(inc)

    @staticmethod
    def wrap_phase(phase):
        """Wrap phase into [-pi, pi)."""
        return (phase + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def generate_synthetic_gunw(shape=(512, 512), seed: int = 42):
        """Generate a synthetic GUNW-like product.

        Returns a dict with:
            unwrapped_phase, wrapped_phase, coherence_before, coherence_after,
            amplitude, dem, truth_mask, anomalies (list of zone params)
        """
        rng = np.random.default_rng(seed)
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

        # Smooth terrain (DEM, metres)
        dem = (
            800
            + 400 * np.sin(xx / w * 2.5 * np.pi) * np.cos(yy / h * 1.8 * np.pi)
            + 250 * np.exp(-(((xx - 0.7 * w) ** 2 + (yy - 0.3 * h) ** 2) / (2 * (0.2 * w) ** 2)))
        )

        # Background phase: gentle ramp (orbital residual) + atmospheric noise
        phase = 0.6 * (xx / w) - 0.4 * (yy / h)
        atmo = rng.normal(0, 1, shape)
        from scipy.ndimage import gaussian_filter

        phase += 0.5 * gaussian_filter(atmo, sigma=25) * 10

        coh_before = np.clip(0.85 + 0.05 * rng.normal(0, 1, shape), 0, 1)
        coh_before = gaussian_filter(coh_before, 2)
        coh_after = coh_before - np.abs(0.04 * rng.normal(0, 1, shape))

        truth = np.zeros(shape, dtype=bool)
        anomalies = []
        n_zones = int(rng.integers(3, 6))
        centers = []
        for _ in range(n_zones):
            # Rejection-sample centres so zones stay separate
            for _try in range(200):
                cy = rng.uniform(0.12, 0.88) * h
                cx = rng.uniform(0.12, 0.88) * w
                if all(np.hypot(cy - a, cx - b) > 0.22 * min(h, w) for a, b in centers):
                    break
            centers.append((cy, cx))
            sy = rng.uniform(0.025, 0.06) * h
            sx = rng.uniform(0.025, 0.06) * w
            # Subsidence: negative phase. -30 rad ~ -570 mm LOS at L-band
            amp = -rng.uniform(8.0, 30.0)
            g = np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
            phase += amp * g
            coh_after -= rng.uniform(0.35, 0.6) * g
            truth |= g > 0.35
            anomalies.append(dict(cy=cy, cx=cx, sy=sy, sx=sx, phase_amp=amp))

        coh_after = np.clip(gaussian_filter(coh_after, 1), 0, 1)

        # Amplitude: Rayleigh speckle modulated by terrain slope
        gy, gx = np.gradient(dem)
        slope = np.hypot(gx, gy)
        backscatter = 0.5 + 0.5 * (slope / slope.max())
        amplitude = backscatter * rng.rayleigh(1.0, shape)
        amplitude = gaussian_filter(amplitude, 1.2)
        amplitude = (amplitude - amplitude.min()) / (np.ptp(amplitude) + 1e-9)

        return {
            "unwrapped_phase": phase.astype(np.float32),
            "wrapped_phase": SARPhysics.wrap_phase(phase).astype(np.float32),
            "coherence_before": coh_before.astype(np.float32),
            "coherence_after": coh_after.astype(np.float32),
            "amplitude": amplitude.astype(np.float32),
            "dem": dem.astype(np.float32),
            "truth_mask": truth,
            "anomalies": anomalies,
        }
