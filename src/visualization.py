"""Map and chart builders for the Streamlit UI."""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

from .coherence import CoherenceAnalyzer


class DisasterMap:
    def __init__(self, map_style: str = pdk.map_styles.CARTO_DARK):
        self.map_style = map_style

    @staticmethod
    def _grid_lonlat(shape, bounds, step):
        lat_min, lat_max, lon_min, lon_max = bounds
        h, w = shape
        rows = np.arange(0, h, step)
        cols = np.arange(0, w, step)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        lat = lat_max - rr / (h - 1) * (lat_max - lat_min)
        lon = lon_min + cc / (w - 1) * (lon_max - lon_min)
        return rr, cc, lat, lon

    def create_3d_map(self, displacement_map, damage_mask, dem, lat_lon_bounds,
                      events: dict | None = None, heat_step: int = 6) -> pdk.Deck:
        """3D map: heatmap of |displacement|, columns on disaster zones, risk polygons."""
        lat_min, lat_max, lon_min, lon_max = lat_lon_bounds
        disp = np.asarray(displacement_map)
        shape = disp.shape

        # Heatmap of displacement magnitude
        rr, cc, lat, lon = self._grid_lonlat(shape, lat_lon_bounds, heat_step)
        mag = np.abs(disp[rr, cc])
        keep = mag > np.percentile(mag, 40)
        heat_data = [
            {"lon": float(a), "lat": float(b), "w": float(m)}
            for a, b, m in zip(lon[keep], lat[keep], mag[keep])
        ]
        heat = pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position=["lon", "lat"],
            get_weight="w",
            radius_pixels=40,
            intensity=1.0,
            threshold=0.05,
            opacity=0.55,
            color_range=[
                [30, 60, 120], [50, 130, 190], [90, 200, 200],
                [250, 220, 90], [250, 140, 50], [230, 50, 50],
            ],
        )

        # 3D columns on damaged pixels (subsampled)
        col_step = max(2, heat_step // 2)
        rr2, cc2, lat2, lon2 = self._grid_lonlat(shape, lat_lon_bounds, col_step)
        dm = np.asarray(damage_mask)[rr2, cc2]
        d = disp[rr2, cc2]
        z = np.asarray(dem)[rr2, cc2]
        dmax = np.abs(d[dm]).max() if dm.any() else 1.0
        col_data = []
        for a, b, v, e in zip(lon2[dm], lat2[dm], d[dm], z[dm]):
            t = min(abs(v) / dmax, 1.0)
            col_data.append({
                "lon": float(a), "lat": float(b),
                "disp": round(float(v), 1), "dem": round(float(e)),
                "height": float(abs(v)),
                "color": [int(255), int(200 - 170 * t), int(60 - 40 * t), 220],
            })
        columns = pdk.Layer(
            "ColumnLayer",
            data=col_data,
            get_position=["lon", "lat"],
            get_elevation="height",
            elevation_scale=12,
            radius=120,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        layers = [heat, columns]

        # Risk-level polygons
        if events and events.get("features"):
            poly_data = []
            for f in events["features"]:
                p = f["properties"]
                rgb = CoherenceAnalyzer.level_color(p.get("risk_level", "LOW"))
                poly_data.append({
                    "polygon": f["geometry"]["coordinates"][0],
                    "fill": rgb + [60],
                    "line": rgb + [255],
                    "event_id": p.get("event_id"),
                    "risk_level": p.get("risk_level"),
                    "risk_score": p.get("risk_score"),
                    "disp": p.get("max_subsidence_mm"),
                    "dem": "",
                })
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=poly_data,
                get_polygon="polygon",
                get_fill_color="fill",
                get_line_color="line",
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=True,
            ))

        view = pdk.ViewState(
            latitude=(lat_min + lat_max) / 2,
            longitude=(lon_min + lon_max) / 2,
            zoom=10,
            pitch=45,
            bearing=0,
        )
        tooltip = {
            "html": "<b>{event_id}</b> {risk_level} {risk_score}<br/>"
                    "Displacement: <b>{disp}</b> mm<br/>Elevation: {dem} m",
            "style": {"backgroundColor": "#1a1a2e", "color": "white"},
        }
        return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto",
                        map_style=self.map_style, tooltip=tooltip)

    @staticmethod
    def create_displacement_timeline(dates, displacements: dict) -> go.Figure:
        """dates: list of dates; displacements: {series_name: [values_mm]}."""
        fig = go.Figure()
        palette = ["#FF4B4B", "#ff8c42", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
        for i, (name, vals) in enumerate(displacements.items()):
            fig.add_trace(go.Scatter(
                x=list(dates), y=list(vals), mode="lines+markers", name=name,
                line=dict(width=2.5, color=palette[i % len(palette)]),
                marker=dict(size=6),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} mm<extra>" + name + "</extra>",
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Cumulative LOS displacement per event",
            xaxis_title="Acquisition date",
            yaxis_title="Displacement (mm)",
            legend=dict(orientation="h", y=-0.2),
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    @staticmethod
    def create_risk_panel(events) -> str:
        """HTML table of events with colour-coded ● markers."""
        feats = events.get("features", []) if isinstance(events, dict) else events
        if not feats:
            return "<p style='color:#aaa'>No events detected.</p>"
        rows = []
        for f in feats:
            p = f["properties"] if "properties" in f else f
            lvl = p.get("risk_level", "LOW")
            rgb = CoherenceAnalyzer.level_color(lvl)
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(p.get('event_id', '')))}</td>"
                f"<td><span style='color:rgb({rgb[0]},{rgb[1]},{rgb[2]});font-weight:700'>"
                f"{CoherenceAnalyzer.level_emoji(lvl)} {lvl}</span></td>"
                f"<td>{p.get('risk_score', 0):.1f}</td>"
                f"<td>{p.get('lat', '—')}, {p.get('lon', '—')}</td>"
                f"<td>{p.get('area_km2', '—')}</td>"
                f"<td>{p.get('max_subsidence_mm', '—')}</td>"
                f"<td>{p.get('mean_delta_coh', '—')}</td>"
                "</tr>"
            )
        return (
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<thead><tr style='border-bottom:1px solid rgba(255,75,75,0.4);text-align:left'>"
            "<th>Event</th><th>Level</th><th>Score</th><th>Lat, Lon</th>"
            "<th>Area km²</th><th>Max subsidence mm</th><th>Δ coherence</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
