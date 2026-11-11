# NISAR Disaster Detection Platform

Streamlit app that turns NISAR GUNW interferograms into landslide and ground-subsidence
risk maps: phase → displacement, coherence-change analysis, U-Net segmentation,
3D PyDeck maps and risk scoring. Runs fully in demo mode on synthetic data.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Built for NASA Space Apps Challenge 2025.
