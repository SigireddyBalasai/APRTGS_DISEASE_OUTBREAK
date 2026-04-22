#!/usr/bin/env python3
"""Load trained SpatiotemporalHawkes weights and compute intensity for a single disease index.

Usage example:
    python inference_single_disease.py \
      --weights checkpoints/hawkes_weights \
      --csv data/outbreak_dummy.csv \
      --disease 0 --lat 17.3850 --lon 78.4867

If you saved weights with `model.save_weights('checkpoints/hawkes_weights')` use the same path here.
"""

import argparse
import numpy as np
import pandas as pd
from main2 import SpatiotemporalHawkes, preprocess


def season_features_from_days(days: float) -> np.ndarray:
    day_of_year = (days % 365.25) + 1.0
    month_of_year = ((days % 365.25) / 365.25) * 12.0 + 1.0
    day_of_week = (days % 7.0)
    annual_sin = np.sin(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
    annual_cos = np.cos(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
    monthly_sin = np.sin(2.0 * np.pi * (month_of_year - 1.0) / 12.0)
    monthly_cos = np.cos(2.0 * np.pi * (month_of_year - 1.0) / 12.0)
    weekly_sin = np.sin(2.0 * np.pi * day_of_week / 7.0)
    weekly_cos = np.cos(2.0 * np.pi * day_of_week / 7.0)
    return np.array(
        [annual_sin, annual_cos, monthly_sin, monthly_cos, weekly_sin, weekly_cos],
        dtype=np.float32,
    )


def predict_intensity_for_location(
    weights_path: str,
    csv_path: str,
    disease_idx: int,
    lat: float,
    lon: float,
    t_query: float | None = None,
    history_window: int = 500,
) -> float:
    df = pd.read_csv(csv_path)
    data = preprocess(df)
    data["pincode"] = df["pincode"].values if "pincode" in df.columns else np.zeros(len(df))
    area = (
        (data["lat_events"].max() - data["lat_events"].min() + 1)
        * (data["lon_events"].max() - data["lon_events"].min() + 1)
    )

    model = SpatiotemporalHawkes(num_diseases=len(data["types"]), history_window=history_window)
    model.set_context(data, area)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Warning: failed to load weights from {weights_path}: {e}")

    # Parameters
    mu = float(model.mu.numpy())
    beta = float(model.beta.numpy())
    gamma = float(model.gamma.numpy())
    alpha_mat = model.alpha.numpy()
    spatial_w = model.spatial_weights.numpy()
    seasonal_w = model.seasonal_weights.numpy()

    # Historical events
    hist_t = np.asarray(data["t_events"]).astype(np.float32)
    hist_la = np.asarray(data["lat_events"]).astype(np.float32)
    hist_lo = np.asarray(data["lon_events"]).astype(np.float32)
    hist_d = np.asarray(data["disease_events"]).astype(np.int32)
    hist_m = np.asarray(data["marks"]).astype(np.float32)

    if t_query is None:
        t_query = float(data["T"])  # default: end of dataset

    # distances (km) between query point and all historical events
    dr = model.haversine(
        np.array([lat], dtype="float32"),
        np.array([lon], dtype="float32"),
        hist_la,
        hist_lo,
    ).numpy().reshape(-1)

    # time differences (days)
    dt = np.maximum(t_query - hist_t, 0.0)

    # disease-specific alpha for each historical event (alpha[disease_idx, hist_d[i]])
    alpha_for_hist = alpha_mat[disease_idx, hist_d]

    # triggering contribution from history
    trig = np.sum(hist_m * alpha_for_hist * np.exp(-beta * dt) * np.exp(-gamma * (dr ** 2)))

    # spatial + seasonal background
    z_la = (lat - data["lat_mean"]) / (data["lat_std"] + 1e-6)
    z_lo = (lon - data["lon_mean"]) / (data["lon_std"] + 1e-6)
    spatial_log = spatial_w[0] * z_la + spatial_w[1] * z_lo + spatial_w[2] * z_la * z_lo

    season_feats = season_features_from_days(t_query)
    seasonal_factor = float(np.exp(np.dot(seasonal_w, season_feats)))

    intensity = mu * seasonal_factor * np.exp(spatial_log) + float(trig)
    return float(intensity)


def _cli():
    p = argparse.ArgumentParser(description="Inference: single-disease intensity")
    p.add_argument("--weights", default="", help="Path to saved weights (use model.save_weights(...))")
    p.add_argument("--csv", default="data/outbreak_dummy.csv", help="CSV with historical events")
    p.add_argument("--disease", type=int, required=True, help="Disease index (0-based)")
    p.add_argument("--lat", type=float, required=True, help="Latitude of query point")
    p.add_argument("--lon", type=float, required=True, help="Longitude of query point")
    p.add_argument("--t", type=float, default=None, help="Query time in days since dataset start (optional)")
    args = p.parse_args()

    lam = predict_intensity_for_location(
        args.weights, args.csv, args.disease, args.lat, args.lon, t_query=args.t
    )
    print(f"Predicted intensity for disease {args.disease} at ({args.lat},{args.lon}): {lam:.6f}")


if __name__ == "__main__":
    _cli()
