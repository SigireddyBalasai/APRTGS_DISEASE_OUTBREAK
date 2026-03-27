import tensorflow as tf
import numpy as np
import pandas as pd


# ===========================================================================
#  1. MODEL DEFINITION
# ===========================================================================

class SpatiotemporalHawkes(tf.keras.Model):

    def __init__(
        self,
        mu_init=0.5,
        alpha_init=0.3,
        beta_init=1.0,
        gamma_init=0.1,
        lat_bounds=None,
        lon_bounds=None,
        time_grid_size=50,
        space_grid_size=10,
        num_diseases=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mu_init = mu_init
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._gamma_init = gamma_init
        self._lat_bounds = lat_bounds
        self._lon_bounds = lon_bounds
        self._time_grid_size = time_grid_size
        self._space_grid_size = space_grid_size
        self._num_diseases = num_diseases

        self._log_mu = self.add_weight(
            name="log_mu", shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_softplus(mu_init)),
            trainable=True)
        self._log_alpha = self.add_weight(
            name="log_alpha", shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_softplus(alpha_init)),
            trainable=True)
        self._log_beta = self.add_weight(
            name="log_beta", shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_softplus(beta_init)),
            trainable=True)
        self._log_gamma = self.add_weight(
            name="log_gamma", shape=(),
            initializer=tf.keras.initializers.Constant(
                self._inv_softplus(gamma_init)),
            trainable=True)

        # Cross-disease excitation matrix (num_diseases x num_diseases)
        self.cross_alpha = self.add_weight(
            name="cross_alpha",
            shape=(self._num_diseases, self._num_diseases),
            initializer=tf.keras.initializers.Constant(self._inv_softplus(alpha_init)),
            trainable=True)

    @staticmethod
    def _inv_softplus(x):
        x = float(x)
        if x <= 0:
            raise ValueError(f"Initial value must be > 0, got {x}")
        if x > 20.0:
            return x
        return float(np.log(np.exp(x) - 1.0))

    @property
    def mu(self):
        return tf.math.softplus(self._log_mu)

    @property
    def alpha(self):
        return tf.math.softplus(self._log_alpha)

    @property
    def beta(self):
        return tf.math.softplus(self._log_beta)

    @property
    def gamma(self):
        return tf.math.softplus(self._log_gamma)

    @property
    def branching_ratio(self):
        return self.alpha / self.beta

    @staticmethod
    def haversine_distances(lat1, lon1, lat2, lon2):
        R = tf.constant(6371.0, dtype=tf.float32)
        deg2rad = tf.constant(np.pi / 180.0, dtype=tf.float32)
        lat1_r = tf.cast(lat1, tf.float32) * deg2rad
        lat2_r = tf.cast(lat2, tf.float32) * deg2rad
        dlat = (tf.cast(lat2, tf.float32) - tf.cast(lat1, tf.float32)) * deg2rad
        dlon = (tf.cast(lon2, tf.float32) - tf.cast(lon1, tf.float32)) * deg2rad
        a = (tf.math.sin(dlat / 2.0) ** 2
             + tf.math.cos(lat1_r) * tf.math.cos(lat2_r)
             * tf.math.sin(dlon / 2.0) ** 2)
        a = tf.clip_by_value(a, 0.0, 1.0)
        return R * 2.0 * tf.math.asin(tf.math.sqrt(a))

    def _pairwise_haversine(self, lat1, lon1, lat2, lon2):
        return self.haversine_distances(
            tf.expand_dims(lat1, 1), tf.expand_dims(lon1, 1),
            tf.expand_dims(lat2, 0), tf.expand_dims(lon2, 0))

    def call(self, inputs, training=False, mask=None):
        t_ev = tf.cast(tf.reshape(inputs["t_events"], [-1]), tf.float32)
        lat_ev = tf.cast(tf.reshape(inputs["lat_events"], [-1]), tf.float32)
        lon_ev = tf.cast(tf.reshape(inputs["lon_events"], [-1]), tf.float32)
        disease_ev = tf.cast(tf.reshape(inputs["disease_events"], [-1]), tf.int32)
        T = tf.cast(tf.reshape(inputs["T"], []), tf.float32)
        nll = self._neg_log_likelihood(t_ev, lat_ev, lon_ev, disease_ev, T)
        return nll

    def _neg_log_likelihood(self, t_events, lat_events, lon_events, disease_events, T):
        mu = self.mu
        beta = self.beta
        gamma = self.gamma
        n = tf.shape(t_events)[0]
        num_diseases = self._num_diseases
        cross_alpha = tf.math.softplus(self.cross_alpha)  # shape [num_diseases, num_diseases]

        idx = tf.argsort(t_events)
        t_sort = tf.gather(t_events, idx)
        la_sort = tf.gather(lat_events, idx)
        lo_sort = tf.gather(lon_events, idx)
        d_sort = tf.gather(disease_events, idx)

        # TERM 1
        dt_mat = tf.expand_dims(t_sort, 1) - tf.expand_dims(t_sort, 0)
        mask = (tf.linalg.band_part(
            tf.ones([n, n], dtype=tf.float32), -1, 0)
            - tf.eye(n, dtype=tf.float32))
        temporal = tf.math.exp(-beta * tf.maximum(dt_mat, 0.0)) * mask
        dist_mat = self._pairwise_haversine(
            la_sort, lo_sort, la_sort, lo_sort)
        spatial = tf.math.exp(-gamma * dist_mat)

        # Cross-disease excitation: for each event i, sum over all j < i
        # cross_alpha[d_i, d_j] * temporal[i, j] * spatial[i, j]
        d_i = tf.expand_dims(d_sort, 1)  # shape [n, 1]
        d_j = tf.expand_dims(d_sort, 0)  # shape [1, n]
        d_i_b = tf.broadcast_to(d_i, [n, n])  # shape [n, n]
        d_j_b = tf.broadcast_to(d_j, [n, n])  # shape [n, n]
        cross = tf.gather_nd(cross_alpha, tf.stack([d_i_b, d_j_b], axis=-1))  # [n, n]
        trigger_at_events = tf.reduce_sum(cross * temporal * spatial, axis=1)
        lam_events = mu + trigger_at_events
        term1 = tf.reduce_sum(tf.math.log(tf.maximum(lam_events, 1e-10)))

        # TERM 2: Integral over space and time
        la_min, la_max, lo_min, lo_max = self._spatial_bounds(
            la_sort, lo_sort)
        m_t = self._time_grid_size
        m_s = self._space_grid_size

        time_grid = tf.linspace(tf.constant(0.0), T, m_t)
        lat_grid = tf.linspace(la_min, la_max, m_s)
        lon_grid = tf.linspace(lo_min, lo_max, m_s)

        dt_vol = T / tf.cast(m_t - 1, tf.float32)
        dla_vol = (la_max - la_min) / tf.maximum(
            tf.cast(m_s - 1, tf.float32), 1.0)
        dlo_vol = (lo_max - lo_min) / tf.maximum(
            tf.cast(m_s - 1, tf.float32), 1.0)

        la_mesh, lo_mesh = tf.meshgrid(lat_grid, lon_grid, indexing="ij")
        la_flat = tf.reshape(la_mesh, [-1])
        lo_flat = tf.reshape(lo_mesh, [-1])

        S = tf.math.exp(
            -gamma * self._pairwise_haversine(
                la_flat, lo_flat, la_sort, lo_sort))

        tg = tf.expand_dims(time_grid, 1)
        te = tf.expand_dims(t_sort, 0)
        dt_grid = tg - te
        D = tf.math.exp(-beta * tf.maximum(dt_grid, 0.0)) * tf.cast(
            dt_grid > 0.0, tf.float32)

        trigger_grid = tf.matmul(D, tf.transpose(S))
        lam_grid = mu + tf.reduce_sum(trigger_grid, axis=1)
        integral = tf.reduce_sum(lam_grid) * dt_vol * dla_vol * dlo_vol

        return -(term1 - integral)

    def _spatial_bounds(self, lat, lon):
        if self._lat_bounds is not None:
            la_min = tf.constant(self._lat_bounds[0], tf.float32)
            la_max = tf.constant(self._lat_bounds[1], tf.float32)
        else:
            la_min = tf.reduce_min(lat) - 1.0
            la_max = tf.reduce_max(lat) + 1.0
        if self._lon_bounds is not None:
            lo_min = tf.constant(self._lon_bounds[0], tf.float32)
            lo_max = tf.constant(self._lon_bounds[1], tf.float32)
        else:
            lo_min = tf.reduce_min(lon) - 1.0
            lo_max = tf.reduce_max(lon) + 1.0
        return la_min, la_max, lo_min, lo_max

    def intensity_at_points(self, t_query, lat_query, lon_query,
                            t_events, lat_events, lon_events):
        tq = tf.cast(tf.reshape(t_query, [-1]), tf.float32)
        laq = tf.cast(tf.reshape(lat_query, [-1]), tf.float32)
        loq = tf.cast(tf.reshape(lon_query, [-1]), tf.float32)
        te = tf.cast(tf.reshape(t_events, [-1]), tf.float32)
        lae = tf.cast(tf.reshape(lat_events, [-1]), tf.float32)
        loe = tf.cast(tf.reshape(lon_events, [-1]), tf.float32)

        dt = tf.expand_dims(tq, 1) - tf.expand_dims(te, 0)
        D = tf.math.exp(-self.beta * tf.maximum(dt, 0.0)) * tf.cast(
            dt > 0.0, tf.float32)
        S = tf.math.exp(
            -self.gamma * self._pairwise_haversine(laq, loq, lae, loe))
        return self.mu + self.alpha * tf.reduce_sum(D * S, axis=1)

    def predict_risk_map(self, t_query, lat_grid, lon_grid,
                         t_events, lat_events, lon_events):
        la_m, lo_m = tf.meshgrid(lat_grid, lon_grid, indexing="ij")
        la_f = tf.reshape(la_m, [-1])
        lo_f = tf.reshape(lo_m, [-1])
        t_f = tf.fill(tf.shape(la_f), tf.cast(t_query, tf.float32))
        intensities = self.intensity_at_points(
            t_f, la_f, lo_f, t_events, lat_events, lon_events)
        return tf.reshape(intensities, tf.shape(la_m))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "mu_init": float(self.mu.numpy()),
            "alpha_init": float(self.alpha.numpy()),
            "beta_init": float(self.beta.numpy()),
            "gamma_init": float(self.gamma.numpy()),
            "lat_bounds": self._lat_bounds,
            "lon_bounds": self._lon_bounds,
            "time_grid_size": self._time_grid_size,
            "space_grid_size": self._space_grid_size,
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


# ===========================================================================
#  2. TRAINER
# ===========================================================================

class HawkesTrainer:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            learning_rate=0.01)
        self.loss_history = []

    def _train_step(self, t_ev, la_ev, lo_ev, disease_ev, T):
        with tf.GradientTape() as tape:
            nll = self.model(
                {"t_events": t_ev, "lat_events": la_ev,
                 "lon_events": lo_ev, "disease_events": disease_ev, "T": T},
                training=True)

        trainable_vars = self.model.trainable_variables
        if len(trainable_vars) == 0:
            raise RuntimeError("No trainable variables found!")

        grads = tape.gradient(nll, trainable_vars)
        valid_pairs = []
        for g, v in zip(grads, trainable_vars):
            if g is not None:
                g = tf.clip_by_value(g, -10.0, 10.0)
                valid_pairs.append((g, v))

        if len(valid_pairs) == 0:
            raise RuntimeError("All gradients are None!")

        self.optimizer.apply_gradients(valid_pairs)
        return nll

    def fit(self, t_events, lat_events, lon_events, disease_events, T,
            epochs=300, verbose=True):
        t_ev = tf.constant(t_events, dtype=tf.float32)
        la_ev = tf.constant(lat_events, dtype=tf.float32)
        lo_ev = tf.constant(lon_events, dtype=tf.float32)
        disease_ev = tf.constant(disease_events, dtype=tf.int32)
        T_c = tf.constant(T, dtype=tf.float32)

        print(f"\n  Trainable variables: {len(self.model.trainable_variables)}")
        for v in self.model.trainable_variables:
            arr = v.numpy()
            if arr.shape == ():
                # Scalar
                print(f"    {v.name} = {float(arr):.4f}")
            else:
                arr_flat = arr.flatten()
                arr_preview = np.array2string(arr_flat, precision=4, separator=', ', threshold=5)
                print(f"    {v.name}: shape={arr.shape}, values={arr_preview}")
        print()

        for epoch in range(epochs):
            nll = float(self._train_step(t_ev, la_ev, lo_ev, T_c).numpy())
            self.loss_history.append(nll)
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                m = self.model
                print(
                    f"Epoch {epoch:4d} | NLL {nll:+12.2f} | "
                    f"mu={float(m.mu.numpy()):.4f} "
                    f"alpha={float(m.alpha.numpy()):.4f} "
                    f"beta={float(m.beta.numpy()):.4f} "
                    f"gamma={float(m.gamma.numpy()):.4f} "
                    f"ratio={float(m.branching_ratio.numpy()):.4f}")
        return self.loss_history


# ===========================================================================
#  3. DATA PREPROCESSING
# ===========================================================================

def preprocess_disease_data(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    t_ref = df["time"].min()
    df["t_hours"] = (df["time"] - t_ref).dt.total_seconds() / 3600.0

    # Encode disease type as integer
    disease_types, disease_type_indices = np.unique(df["snomet_id"], return_inverse=True)
    df["disease_type"] = disease_type_indices

    t_events = df["t_hours"].values
    lat_events = df["latitude"].values
    lon_events = df["longitude"].values
    disease_events = df["disease_type"].values.astype(np.int32)
    T = float(t_events.max()) + 1.0

    print(f"Number of events     : {len(t_events)}")
    print(f"Time range           : {df['time'].min()} -> {df['time'].max()}")
    print(f"Time range (hours)   : [0, {T:.1f}]")
    print(f"Latitude range       : [{lat_events.min():.4f}, "
          f"{lat_events.max():.4f}]")
    print(f"Longitude range      : [{lon_events.min():.4f}, "
          f"{lon_events.max():.4f}]")
    print(f"Unique pincodes      : {df['pincode'].nunique()}")
    print(f"Unique locations     : "
          f"{len(df.groupby(['latitude', 'longitude']))}")

    return {
        "t_events": t_events,
        "lat_events": lat_events,
        "lon_events": lon_events,
        "disease_events": disease_events,
        "disease_types": disease_types,
        "T": T,
        "t_ref": t_ref,
        "df_sorted": df,
    }


# ===========================================================================
#  4. HOTSPOT PREDICTOR (with robust risk classification)
# ===========================================================================

def classify_risk(scores):
    """
    Robust risk classification that handles identical/near-identical scores.
    Uses rank-based quartiles instead of value-based pd.cut.
    """
    n = len(scores)
    if n == 0:
        return pd.Series([], dtype="object")

    # Rank from highest to lowest
    ranks = scores.rank(method="first", ascending=False)

    # Assign risk levels based on rank position
    labels = []
    for r in ranks:
        pct = r / n
        if pct <= 0.25:
            labels.append("CRITICAL")
        elif pct <= 0.50:
            labels.append("HIGH")
        elif pct <= 0.75:
            labels.append("MEDIUM")
        else:
            labels.append("LOW")

    return pd.Categorical(labels, categories=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                          ordered=True)


class HotspotPredictor:
    def __init__(self, model, data_dict):
        self.model = model
        self.data = data_dict

    def predict_future_risk_at_locations(self, future_hours_ahead=24.0):
        df = self.data["df_sorted"]
        t_events = self.data["t_events"]
        lat_events = self.data["lat_events"]
        lon_events = self.data["lon_events"]
        t_ref = self.data["t_ref"]

        t_last = float(t_events.max())
        t_future = t_last + future_hours_ahead

        locations = df.groupby("pincode").agg({
            "latitude": "first",
            "longitude": "first",
            "snomet_id": "first",
        }).reset_index()

        lats = locations["latitude"].values.astype(np.float32)
        lons = locations["longitude"].values.astype(np.float32)
        t_query = np.full_like(lats, t_future)

        intensities = self.model.intensity_at_points(
            t_query, lats, lons,
            t_events, lat_events, lon_events).numpy()

        locations["risk_score"] = intensities
        locations["risk_rank"] = locations["risk_score"].rank(
            ascending=False, method="min").astype(int)

        # Robust risk classification (handles identical scores)
        locations["risk_level"] = classify_risk(locations["risk_score"])

        future_datetime = t_ref + pd.Timedelta(hours=t_future)
        locations["prediction_time"] = future_datetime

        # Decay info for context
        beta_val = float(self.model.beta.numpy())
        decay_factor = np.exp(-beta_val * future_hours_ahead)
        locations["temporal_decay"] = decay_factor

        locations = locations.sort_values(
            "risk_score", ascending=False).reset_index(drop=True)
        return locations

    def predict_risk_at_current(self):
        """
        Predict risk at the LAST event time (where triggering is still active).
        This shows meaningful spatial variation.
        """
        df = self.data["df_sorted"]
        t_events = self.data["t_events"]
        lat_events = self.data["lat_events"]
        lon_events = self.data["lon_events"]
        t_ref = self.data["t_ref"]

        t_last = float(t_events.max())

        locations = df.groupby("pincode").agg({
            "latitude": "first",
            "longitude": "first",
            "snomet_id": "first",
        }).reset_index()

        lats = locations["latitude"].values.astype(np.float32)
        lons = locations["longitude"].values.astype(np.float32)
        t_query = np.full_like(lats, t_last)

        intensities = self.model.intensity_at_points(
            t_query, lats, lons,
            t_events, lat_events, lon_events).numpy()

        locations["risk_score"] = intensities
        locations["risk_rank"] = locations["risk_score"].rank(
            ascending=False, method="min").astype(int)
        locations["risk_level"] = classify_risk(locations["risk_score"])

        eval_datetime = t_ref + pd.Timedelta(hours=t_last)
        locations["evaluation_time"] = eval_datetime
        locations = locations.sort_values(
            "risk_score", ascending=False).reset_index(drop=True)
        return locations

    def predict_risk_multi_horizon(self, horizons_hours=None):
        """
        Predict risk at multiple future time horizons.
        Shows how risk decays over time per location.
        """
        if horizons_hours is None:
            horizons_hours = [0, 1, 3, 6, 12, 24, 48]

        df = self.data["df_sorted"]
        t_events = self.data["t_events"]
        lat_events = self.data["lat_events"]
        lon_events = self.data["lon_events"]
        t_ref = self.data["t_ref"]

        t_last = float(t_events.max())

        locations = df.groupby("pincode").agg({
            "latitude": "first",
            "longitude": "first",
            "snomet_id": "first",
        }).reset_index()

        lats = locations["latitude"].values.astype(np.float32)
        lons = locations["longitude"].values.astype(np.float32)

        results = []
        for h in horizons_hours:
            t_q = t_last + h
            t_query = np.full_like(lats, t_q)
            intensities = self.model.intensity_at_points(
                t_query, lats, lons,
                t_events, lat_events, lon_events).numpy()

            for i, row in locations.iterrows():
                results.append({
                    "pincode": row["pincode"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "hours_ahead": h,
                    "prediction_time": t_ref + pd.Timedelta(hours=t_q),
                    "risk_score": intensities[i],
                })

        result_df = pd.DataFrame(results)
        return result_df

    def predict_risk_grid(self, future_hours_ahead=0.0, grid_res=50):
        t_events = self.data["t_events"]
        lat_events = self.data["lat_events"]
        lon_events = self.data["lon_events"]

        t_future = float(t_events.max()) + future_hours_ahead

        lat_min = float(lat_events.min()) - 0.5
        lat_max = float(lat_events.max()) + 0.5
        lon_min = float(lon_events.min()) - 0.5
        lon_max = float(lon_events.max()) + 0.5

        lat_grid = tf.linspace(lat_min, lat_max, grid_res)
        lon_grid = tf.linspace(lon_min, lon_max, grid_res)

        risk_map = self.model.predict_risk_map(
            t_future, lat_grid, lon_grid,
            t_events, lat_events, lon_events).numpy()

        return lat_grid.numpy(), lon_grid.numpy(), risk_map

    def predict_temporal_evolution(self, pincode, hours_ahead=72, step_hours=1.0):
        df = self.data["df_sorted"]
        t_events = self.data["t_events"]
        lat_events = self.data["lat_events"]
        lon_events = self.data["lon_events"]
        t_ref = self.data["t_ref"]

        # Ensure pincode is string for comparison
        pincode_str = str(pincode)
        df_pincode = df[df["pincode"] == pincode_str]
        if df_pincode.empty:
            print(f"Warning: pincode {pincode} not found in data. Skipping.")
            return pd.DataFrame()
        loc = df_pincode.iloc[0]
        lat = np.float32(loc["latitude"])
        lon = np.float32(loc["longitude"])

        t_last = float(t_events.max())
        time_points = np.arange(
            t_last, t_last + hours_ahead, step_hours).astype(np.float32)

        lats = np.full_like(time_points, lat)
        lons = np.full_like(time_points, lon)

        intensities = self.model.intensity_at_points(
            time_points, lats, lons,
            t_events, lat_events, lon_events).numpy()

        return pd.DataFrame({
            "time": [t_ref + pd.Timedelta(hours=float(h))
                     for h in time_points],
            "hours_from_last": time_points - t_last,
            "intensity": intensities,
        })

    def find_hotspot_clusters(self, future_hours_ahead=0.0,
                              grid_res=50, top_k=5):
        lat_grid, lon_grid, risk_map = self.predict_risk_grid(
            future_hours_ahead, grid_res)
        la_m, lo_m = np.meshgrid(lat_grid, lon_grid, indexing="ij")
        flat_lat = la_m.ravel()
        flat_lon = lo_m.ravel()
        flat_risk = risk_map.ravel()
        top_idx = np.argsort(flat_risk)[::-1][:top_k]
        return pd.DataFrame({
            "latitude": flat_lat[top_idx],
            "longitude": flat_lon[top_idx],
            "intensity": flat_risk[top_idx],
            "rank": range(1, top_k + 1),
        })


# ===========================================================================
#  5. MAIN PIPELINE
# ===========================================================================

def main():


    # --- Load Data ---
    outbreak_df = pd.read_csv("data/outbreak_dummy.csv")
    pincode_df = pd.read_csv("data/pincode.csv")
    outbreak_df["pincode"] = outbreak_df["pincode"].astype(str)
    pincode_df["pincode"] = pincode_df["pincode"].astype(str)

    # Average lat/long for each pincode (ignoring NA)
    pincode_avg = pincode_df.copy()
    pincode_avg["latitude"] = pd.to_numeric(pincode_avg["latitude"], errors="coerce")
    pincode_avg["longitude"] = pd.to_numeric(pincode_avg["longitude"], errors="coerce")
    pincode_avg = pincode_avg[
        pincode_avg["latitude"].notna() & pincode_avg["longitude"].notna()
    ]
    pincode_avg = pincode_avg.groupby("pincode", as_index=False)[["latitude", "longitude"]].mean()
    merged = pd.merge(outbreak_df, pd.DataFrame(pincode_avg), on="pincode", how="left")

    # Remove rows with missing latitude/longitude after merge
    before_drop = len(merged)
    merged = merged.dropna(subset=["latitude", "longitude", "time"])
    after_drop = len(merged)
    if after_drop < before_drop:
        print(f"Dropped {before_drop - after_drop} rows with missing lat/lon/time after merge.")

    print("Merged Outbreak Data with Lat/Long (Averaged):")
    print(merged.head(10))

    # --- Data Preparation ---
    df = merged.copy()
    df["time"] = pd.to_datetime(df["time"])
    # If snomet_id is not string, convert
    df["snomet_id"] = df["snomet_id"].astype(str)
    # Optionally add gender and case_id for compatibility
    if "gender" not in df.columns:
        df["gender"] = np.random.choice(["M", "F"], size=len(df))
    if "case_id" not in df.columns:
        df["case_id"] = range(len(df))

    print("=" * 70)
    print("  DISEASE HOTSPOT PREDICTION - Spatiotemporal Hawkes Process")
    print("=" * 70)
    print("\nRaw data sample:")
    print(df.head(10))

    # ------------------------------------------------------------------
    # B) Preprocess
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  DATA PREPROCESSING")
    print("-" * 70)
    data_dict = preprocess_disease_data(df)

    print("\n  Note: pincode 504297 at lat=25.2 is far from others "
          "(17.7-19.5).")
    print("  This may be a data entry error.\n")

    # ------------------------------------------------------------------
    # C) Build model
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  MODEL BUILDING")
    print("-" * 70)

    lat_min = float(data_dict["lat_events"].min()) - 0.5
    lat_max = float(data_dict["lat_events"].max()) + 0.5
    lon_min = float(data_dict["lon_events"].min()) - 0.5
    lon_max = float(data_dict["lon_events"].max()) + 0.5

    model = SpatiotemporalHawkes(
        mu_init=0.5,
        alpha_init=0.3,
        beta_init=0.5,
        gamma_init=0.01,
        lat_bounds=(lat_min, lat_max),
        lon_bounds=(lon_min, lon_max),
        time_grid_size=40,
        space_grid_size=8,
        num_diseases=len(data_dict["disease_types"]),
    )

    # Dry run
    dummy_nll = model({
        "t_events": tf.constant(data_dict["t_events"]),
        "lat_events": tf.constant(data_dict["lat_events"]),
        "lon_events": tf.constant(data_dict["lon_events"]),
        "disease_events": tf.constant(data_dict["disease_events"]),
        "T": tf.constant(data_dict["T"]),
    }, training=False)
    print(f"\n  Initial NLL: {float(dummy_nll.numpy()):.2f}")
    print(f"  Trainable variables: {len(model.trainable_variables)}")
    for v in model.trainable_variables:
        print(f"    {v.name}: shape={v.shape}, value={float(v.numpy()):.4f}")
    model.summary()

    # ------------------------------------------------------------------
    # D) Fit
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  MODEL TRAINING (MLE via Adam)")
    print("-" * 70)

    trainer = HawkesTrainer(
        model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    )
    loss_history = trainer.fit(
        data_dict["t_events"],
        data_dict["lat_events"],
        data_dict["lon_events"],
        data_dict["disease_events"],
        T=data_dict["T"],
        epochs=500,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # E) Fitted parameters
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  FITTED PARAMETERS")
    print("-" * 70)
    mu_val = float(model.mu.numpy())
    alpha_val = float(model.alpha.numpy())
    beta_val = float(model.beta.numpy())
    gamma_val = float(model.gamma.numpy())
    br = float(model.branching_ratio.numpy())

    print(f"  mu (background rate) = {mu_val:.6f}")
    print(f"  alpha (excitation)   = {alpha_val:.6f}")
    print(f"  beta (temporal decay)= {beta_val:.6f}")
    print(f"  gamma (spatial decay)= {gamma_val:.6f}")
    print(f"  branching ratio a/b  = {br:.6f}")
    print(f"  Status: {'STABLE' if br < 1 else 'UNSTABLE'}")

    # Show temporal decay at various horizons
    print(f"\n  Temporal decay exp(-beta * h):")
    for h in [0, 1, 3, 6, 12, 24, 48]:
        decay = np.exp(-beta_val * h)
        print(f"    {h:3d}h ahead: decay = {decay:.6f} "
              f"({'significant' if decay > 0.01 else 'negligible'})")

    # ------------------------------------------------------------------
    # F) Predict hotspots
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  HOTSPOT PREDICTIONS")
    print("-" * 70)

    predictor = HotspotPredictor(model, data_dict)

    # F1: Current risk (at last event time) — shows spatial variation
    print("\n>>> CURRENT RISK (at last event time, shows spatial variation):\n")
    risk_now = predictor.predict_risk_at_current()
    print(risk_now[[
        "pincode", "snomet_id", "latitude", "longitude",
        "risk_score", "risk_rank", "risk_level", "evaluation_time",
    ]].to_string(index=False))

    # F2: Multi-horizon forecast
    print("\n>>> MULTI-HORIZON FORECAST (how risk decays over time):\n")
    horizons = [0, 1, 3, 6, 12, 24]
    multi = predictor.predict_risk_multi_horizon(horizons)

    # Pivot for readable output
    pivot = multi.pivot_table(
        index="pincode", columns="hours_ahead",
        values="risk_score").round(6)
    pivot.columns = [f"{h}h" for h in pivot.columns]
    print(pivot.to_string())

    # F3: 1-hour ahead (where triggering is still visible)
    print("\n>>> Risk at known locations (1 hour ahead):\n")
    risk_1h = predictor.predict_future_risk_at_locations(
        future_hours_ahead=1.0)
    print(risk_1h[[
        "pincode", "snomet_id", "latitude", "longitude",
        "risk_score", "risk_rank", "risk_level",
        "prediction_time", "temporal_decay",
    ]].to_string(index=False))

    # F4: 6-hour ahead
    print("\n>>> Risk at known locations (6 hours ahead):\n")
    risk_6h = predictor.predict_future_risk_at_locations(
        future_hours_ahead=6.0)
    print(risk_6h[[
        "pincode", "risk_score", "risk_rank", "risk_level",
        "temporal_decay",
    ]].to_string(index=False))

    # F5: 24-hour ahead
    print("\n>>> Risk at known locations (24 hours ahead):\n")
    risk_24h = predictor.predict_future_risk_at_locations(
        future_hours_ahead=24.0)
    print(risk_24h[[
        "pincode", "risk_score", "risk_rank", "risk_level",
        "temporal_decay",
    ]].to_string(index=False))

    # F6: Top hotspot grid cells at current time
    print("\n>>> Top-5 hotspot grid cells (at current time):\n")
    hotspots = predictor.find_hotspot_clusters(
        future_hours_ahead=0.0, grid_res=40, top_k=5)
    print(hotspots.to_string(index=False))

    # F7: Temporal forecast for top pincode
    top_pincode = int(risk_now.iloc[0]["pincode"])
    print(f"\n>>> Temporal forecast for pincode {top_pincode} "
          f"(next 48h, every 3h):\n")
    temporal = predictor.predict_temporal_evolution(
        top_pincode, hours_ahead=48, step_hours=3.0)
    print(temporal.to_string(index=False))

    # ------------------------------------------------------------------
    # G) Visualisation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  VISUALISATION")
    print("-" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))

        # G1: Training loss
        ax = axes[0, 0]
        ax.plot(loss_history, linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative Log-Likelihood")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

        # G2: Risk map at current time
        ax = axes[0, 1]
        lat_grid, lon_grid, risk_map_now = predictor.predict_risk_grid(
            future_hours_ahead=0.0, grid_res=60)
        im = ax.imshow(
            risk_map_now, origin="lower",
            extent=[lon_grid.min(), lon_grid.max(),
                    lat_grid.min(), lat_grid.max()],
            aspect="auto", cmap="YlOrRd")
        ax.scatter(
            data_dict["lon_events"], data_dict["lat_events"],
            c="blue", s=8, alpha=0.3, label="Events")
        locs = df.groupby("pincode").first().reset_index()
        ax.scatter(
            locs["longitude"], locs["latitude"],
            c="cyan", s=60, marker="^", edgecolors="black",
            linewidths=0.5, label="Locations", zorder=5)
        for _, row in locs.iterrows():
            ax.annotate(
                str(int(row["pincode"])),
                (row["longitude"], row["latitude"]),
                fontsize=5, ha="left", va="bottom",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="black", alpha=0.5))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Risk Map (Current Time)")
        ax.legend(fontsize=7)
        plt.colorbar(im, ax=ax, label="Intensity")

        # G3: Risk map 6h ahead
        ax = axes[0, 2]
        lat_grid6, lon_grid6, risk_map_6h = predictor.predict_risk_grid(
            future_hours_ahead=6.0, grid_res=60)
        im = ax.imshow(
            risk_map_6h, origin="lower",
            extent=[lon_grid6.min(), lon_grid6.max(),
                    lat_grid6.min(), lat_grid6.max()],
            aspect="auto", cmap="YlOrRd")
        ax.scatter(
            locs["longitude"], locs["latitude"],
            c="cyan", s=60, marker="^", edgecolors="black",
            linewidths=0.5, zorder=5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Risk Map (6h Ahead)")
        plt.colorbar(im, ax=ax, label="Intensity")

        # G4: Bar chart of current risk
        ax = axes[1, 0]
        color_map = {"CRITICAL": "red", "HIGH": "orange",
                     "MEDIUM": "gold", "LOW": "green"}
        bar_colors = [color_map.get(str(lvl), "gray")
                      for lvl in risk_now["risk_level"]]
        ax.barh(
            risk_now["pincode"].astype(str),
            risk_now["risk_score"],
            color=bar_colors)
        ax.set_xlabel("Risk Score (Intensity)")
        ax.set_ylabel("Pincode")
        ax.set_title("Risk by Pincode (Current Time)")
        ax.grid(True, alpha=0.3, axis="x")
        for idx_row, row in risk_now.iterrows():
            ax.text(
                row["risk_score"] + 0.0005,
                idx_row,
                f" {row['risk_level']}",
                va="center", fontsize=7, fontweight="bold")

        # G5: Multi-horizon decay for all locations
        ax = axes[1, 1]
        for _, grp in multi.groupby("pincode"):
            pincode = int(grp["pincode"].iloc[0])
            ax.plot(grp["hours_ahead"], grp["risk_score"],
                    marker="o", markersize=3, label=str(pincode))
        ax.axhline(mu_val, color="gray", linestyle="--",
                    label=f"Background mu={mu_val:.4f}")
        ax.set_xlabel("Hours Ahead")
        ax.set_ylabel("Intensity")
        ax.set_title("Risk Decay Over Time (All Locations)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        # G6: Temporal forecast for top pincode
        ax = axes[1, 2]
        temporal_fine = predictor.predict_temporal_evolution(
            top_pincode, hours_ahead=48, step_hours=0.5)
        ax.plot(temporal_fine["hours_from_last"],
                temporal_fine["intensity"],
                color="red", linewidth=1.5)
        ax.axhline(mu_val, color="gray", linestyle="--",
                    label=f"Background mu={mu_val:.4f}")
        ax.set_xlabel("Hours from Last Event")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Temporal Risk - Pincode {top_pincode}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("disease_hotspot_predictions.png", dpi=150,
                    bbox_inches="tight")
        print("\n  Plot saved: disease_hotspot_predictions.png")
        plt.close()
    except ImportError:
        print("\n  matplotlib not available - skipping plots.")

    # ------------------------------------------------------------------
    # H) Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Fitted parameters:")
    print(f"    mu    = {mu_val:.6f}  (background disease rate)")
    print(f"    alpha = {alpha_val:.6f}  (contagion strength)")
    print(f"    beta  = {beta_val:.6f}  (temporal decay per hour)")
    print(f"    gamma = {gamma_val:.6f}  (spatial decay per km)")
    print(f"    ratio = {br:.4f} ({'STABLE' if br < 1 else 'UNSTABLE'})")

    half_life_hours = np.log(2) / beta_val
    half_life_km = np.log(2) / gamma_val if gamma_val > 0 else float("inf")
    print(f"\n  Derived quantities:")
    print(f"    Temporal half-life  : {half_life_hours:.1f} hours")
    print(f"    Spatial half-life   : {half_life_km:.1f} km")

    print(f"\n  Highest risk (current time):")
    for _, row in risk_now.head(3).iterrows():
        print(f"    Pincode {int(row['pincode'])} "
              f"({row['latitude']:.4f}, {row['longitude']:.4f}) "
              f"= {row['risk_score']:.6f} [{row['risk_level']}]")

    print(f"\n  Lowest risk (current time):")
    for _, row in risk_now.tail(3).iterrows():
        print(f"    Pincode {int(row['pincode'])} "
              f"({row['latitude']:.4f}, {row['longitude']:.4f}) "
              f"= {row['risk_score']:.6f} [{row['risk_level']}]")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()