import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
outbreak_df = pd.read_csv("data/outbreak_dummy.csv")
pincode_df = pd.read_csv("data/pincode.csv")
outbreak_df["pincode"] = outbreak_df["pincode"].astype(str)
pincode_df["pincode"] = pincode_df["pincode"].astype(str)

# Average lat/long for each pincode (ignoring NA)
pincode_avg = pincode_df.copy()
pincode_avg = pincode_avg[
    pincode_avg["latitude"].notna() & pincode_avg["longitude"].notna()
]
pincode_avg["latitude"] = pd.to_numeric(pincode_avg["latitude"], errors="coerce")
pincode_avg["longitude"] = pd.to_numeric(pincode_avg["longitude"], errors="coerce")
pincode_avg = pincode_avg.groupby("pincode", as_index=False)[
    ["latitude", "longitude"]
].mean()
merged = pd.merge(outbreak_df, pd.DataFrame(pincode_avg), on="pincode", how="left")
print("Merged Outbreak Data with Lat/Long (Averaged):")
print(merged)


# --- Data Preparation ---
df = merged.copy()
df["datetime"] = pd.to_datetime(df["time"])
df["disease_class"] = df["snomet_id"].astype(str)
df["gender"] = np.random.choice(["M", "F"], size=len(df))
df["case_id"] = range(len(df))


def validate_data(dataframe):
    print("\nData Validation Report")
    print(f"Total records: {len(dataframe)}")
    print(f"Date range: {dataframe['datetime'].min()} to {dataframe['datetime'].max()}")
    print(f"Unique diseases: {dataframe['disease_class'].unique()}")
    print(f"Unique pincodes: {dataframe['pincode'].nunique()}")
    print(f"Gender distribution:\n{dataframe['gender'].value_counts()}")
    print(f"Missing values:\n{dataframe.isnull().sum()}")
    duplicates = dataframe.duplicated(subset=["case_id"]).sum()
    print(f"Duplicate records: {duplicates}")
    return dataframe


validate_data(df)


# --- Hawkes Process ---
class HawkesPointProcess:
    def __init__(self, baseline_mu=0.5, alpha=0.3, beta=0.2):
        self.mu = baseline_mu
        self.alpha = alpha
        self.beta = beta
        self.events = []

    def add_events(self, timestamps):
        self.events = list(np.sort(np.array(timestamps)).tolist())

    def intensity(self, t):
        if len(self.events) == 0:
            return self.mu
        past_events = self.events[self.events < t]
        if len(past_events) == 0:
            return self.mu
        time_gaps = t - past_events
        hawkes_term = self.alpha * np.sum(np.exp(-self.beta * time_gaps))
        return self.mu + hawkes_term

    def log_likelihood(self):
        if len(self.events) == 0:
            return 0
        log_lik = 0
        for t in self.events:
            log_lik += np.log(self.intensity(t) + 1e-10)
        t_max = self.events[-1]
        time_grid = np.linspace(0, t_max, 1000)
        dt = t_max / 1000
        integral = 0
        for t in time_grid:
            integral += self.intensity(t) * dt
        return log_lik - integral

    def predict_excess_risk(self, t, window_days=7):
        baseline_expected = self.mu * window_days
        recent_events = self.events[self.events > (t - 1)]
        if len(recent_events) == 0:
            excess_rate = 0
        else:
            excess_rate = (
                self.alpha
                * len(recent_events)
                * (1 - np.exp(-self.beta * window_days))
                / self.beta
            )
        excess_cases = excess_rate * window_days
        excess_risk = excess_cases / baseline_expected if baseline_expected > 0 else 0
        return {
            "baseline_expected": baseline_expected,
            "excess_expected": excess_cases,
            "relative_risk": 1 + excess_risk,
        }


print("\n[2] Hawkes Process Analysis...")
dengue_data = df[df["disease_class"] == "Dengue"].copy()
days_delta = (dengue_data["datetime"] - dengue_data["datetime"].min()).dt.days.values
hawkes = HawkesPointProcess(baseline_mu=0.3, alpha=0.25, beta=0.15)
hawkes.add_events(days_delta)
print(f"Baseline intensity (μ): {hawkes.mu:.4f} cases/day")
print(f"Branching ratio (α): {hawkes.alpha:.4f}")
print(f"Temporal decay (β): {hawkes.beta:.4f}")
print(f"Log-likelihood: {hawkes.log_likelihood():.2f}")
risk_pred = hawkes.predict_excess_risk(dengue_data["days"].max(), window_days=7)
print("\nNext 7-day forecast:")
print(f"  Baseline expected: {risk_pred['baseline_expected']:.2f}")
print(f"  Excess expected: {risk_pred['excess_expected']:.2f}")
print(f"  Relative risk: {risk_pred['relative_risk']:.2f}x")


# --- Scan Statistic ---
class ScanStatistic:
    def __init__(self, scan_df, baseline_period_days=14):
        self.df = scan_df.copy()
        self.baseline_period_days = baseline_period_days
        self.compute_baseline()

    def compute_baseline(self):
        baseline_end = self.df["datetime"].min() + timedelta(
            days=self.baseline_period_days
        )
        baseline_data = self.df[self.df["datetime"] <= baseline_end]
        self.baseline_rate = len(baseline_data) / self.baseline_period_days
        self.pincode_rates = (
            baseline_data.groupby("pincode").size() / self.baseline_period_days
        )
        self.disease_rates = (
            baseline_data.groupby("disease_class").size() / self.baseline_period_days
        )
        print(f"Baseline rate: {self.baseline_rate:.3f} cases/day")

    def get_clusters(self, spatial_radius_km=5, temporal_window_days=7, min_cases=2):
        clusters_local = []
        unique_locations = (
            self.df.groupby("pincode")
            .agg({"latitude": "first", "longitude": "first"})
            .reset_index()
        )
        baseline_end = self.df["datetime"].min() + timedelta(
            days=self.baseline_period_days
        )
        recent_data = self.df[self.df["datetime"] > baseline_end].copy()
        if recent_data.empty:
            print(
                "No recent data after baseline period for cluster detection. Skipping scan statistic."
            )
            return pd.DataFrame()
        date_range = pd.date_range(
            start=recent_data["datetime"].min(),
            end=recent_data["datetime"].max(),
            freq=f"{max(1, temporal_window_days // 2)}D",
        )
        for center_date in date_range:
            window_start = center_date
            window_end = center_date + timedelta(days=temporal_window_days)
            time_window_data = recent_data[
                (recent_data["datetime"] >= window_start)
                & (recent_data["datetime"] <= window_end)
            ]
            if len(time_window_data) < min_cases:
                continue
            for _, loc in unique_locations.iterrows():
                lat, lon = loc["latitude"], loc["longitude"]
                spatial_window_data = time_window_data[
                    self._haversine_distance(
                        time_window_data["latitude"].values,
                        time_window_data["longitude"].values,
                        lat,
                        lon,
                    )
                    <= spatial_radius_km
                ]
                if len(spatial_window_data) < min_cases:
                    continue
                O_w = len(spatial_window_data)
                E_w = (
                    self.baseline_rate
                    * (temporal_window_days / 365)
                    * (spatial_radius_km / 1000) ** 2
                    * len(self.df["datetime"].unique())
                )
                if E_w < 1:
                    E_w = 1
                N = len(time_window_data)
                if O_w > 0 and E_w > 0 and N > O_w:
                    lr = (O_w / E_w) ** O_w * ((N - O_w) / (N - E_w)) ** (N - O_w)
                    log_lr = O_w * np.log(O_w / E_w + 1e-10) + (N - O_w) * np.log(
                        (N - O_w) / (N - E_w) + 1e-10
                    )
                else:
                    lr = 0
                    log_lr = 0
                relative_risk = O_w / E_w if E_w > 0 else 1
                clusters_local.append(
                    {
                        "center_lat": lat,
                        "center_lon": lon,
                        "pincode": loc["pincode"],
                        "window_start": window_start,
                        "window_end": window_end,
                        "observed": O_w,
                        "expected": E_w,
                        "relative_risk": relative_risk,
                        "likelihood_ratio": lr,
                        "log_likelihood_ratio": log_lr,
                        "diseases": spatial_window_data["disease_class"]
                        .value_counts()
                        .to_dict(),
                        "n_males": (spatial_window_data["gender"] == "M").sum(),
                        "n_females": (spatial_window_data["gender"] == "F").sum(),
                    }
                )
        if clusters_local:
            clusters_df = pd.DataFrame(clusters_local)
            clusters_df = clusters_df.sort_values(
                "log_likelihood_ratio", ascending=False
            )
            threshold = clusters_df["log_likelihood_ratio"].quantile(0.95)
            significant = clusters_df[clusters_df["log_likelihood_ratio"] >= threshold]
            return significant
        return pd.DataFrame()

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a_local = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c_local = 2 * np.arcsin(np.sqrt(a_local))
        return R * c_local


print("\n[3] Spatio-Temporal Scan Statistic...")
scanner = ScanStatistic(df, baseline_period_days=3)
clusters = scanner.get_clusters(
    spatial_radius_km=5, temporal_window_days=7, min_cases=2
)
print(f"\nClusters detected: {len(clusters)}")
if len(clusters) > 0:
    print("\nTop clusters:")
    print(
        clusters[
            [
                "pincode",
                "observed",
                "expected",
                "relative_risk",
                "likelihood_ratio",
                "log_likelihood_ratio",
                "window_start",
                "window_end",
                "diseases",
            ]
        ].head(3)
    )


# --- Anomaly Detection ---
class AnomalyDetector:
    def __init__(self, contamination : float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()

    def extract_features(self, anomaly_df, window_days=7):
        features = []
        dates = []
        pincodes = []
        diseases = []
        for (date, pincode, disease), group in anomaly_df.groupby(
            [anomaly_df["datetime"].dt.date, "pincode", "disease_class"]
        ):
            if len(group) < 1:
                continue
            daily_count = len(group)
            date_start = pd.Timestamp(date) - timedelta(days=window_days)
            date_end = pd.Timestamp(date)
            historical = anomaly_df[
                (anomaly_df["datetime"] >= date_start)
                & (anomaly_df["datetime"] < date_end)
                & (anomaly_df["pincode"] == pincode)
                & (anomaly_df["disease_class"] == disease)
            ]
            avg_count = len(historical) / window_days if window_days > 0 else 0
            if avg_count == 0:
                pct_change = 0
            else:
                pct_change = (daily_count - avg_count) / avg_count
            m_count = (group["gender"] == "M").sum()
            f_count = (group["gender"] == "F").sum()
            gender_ratio = m_count / (f_count + 1) if f_count > 0 else m_count
            dow = pd.Timestamp(date).dayofweek
            feature_vector = [daily_count, avg_count, pct_change, gender_ratio, dow]
            features.append(feature_vector)
            dates.append(date)
            pincodes.append(pincode)
            diseases.append(disease)
        if not features:
            return None
        X = np.array(features)
        return {"X": X, "dates": dates, "pincodes": pincodes, "diseases": diseases}

    def detect_anomalies(self, anomaly_df):
        feature_data = self.extract_features(anomaly_df)
        if feature_data is None:
            return None
        X = feature_data["X"]
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        anomalies_local = pd.DataFrame(
            {
                "date": feature_data["dates"],
                "pincode": feature_data["pincodes"],
                "disease": feature_data["diseases"],
                "daily_count": X[:, 0] if X.ndim > 1 else X,
                "anomaly_score": scores,
                "is_anomaly": predictions == -1,
            }
        )
        return anomalies_local.sort_values("anomaly_score")


print("\n[4] Anomaly Detection (Isolation Forest)...")
detector = AnomalyDetector(contamination=0.2)
anomalies = detector.detect_anomalies(df)
if anomalies is not None:
    n_anomalies = (anomalies["is_anomaly"]).sum()
    print(f"Anomalies detected: {n_anomalies}")
    print("\nTop anomalies:")
    print(anomalies[anomalies["is_anomaly"]].head(3))


# --- Alert Report ---
def build_alert_report(cluster_data, anomaly_data):
    alert_report = {
        "timestamp": datetime.now(),
        "n_clusters": len(cluster_data),
        "n_anomalies": (anomaly_data["is_anomaly"].sum() if anomaly_data is not None else 0),
        "top_clusters": [],
        "top_anomalies": [],
    }
    if len(cluster_data) > 0:
        top_clusters = cluster_data.nlargest(3, "log_likelihood_ratio")
        for _, row in top_clusters.iterrows():
            alert_report["top_clusters"].append(
                {
                    "pincode": row["pincode"],
                    "observed": int(row["observed"]),
                    "expected": round(row["expected"], 2),
                    "relative_risk": round(row["relative_risk"], 2),
                    "likelihood_ratio": round(row["likelihood_ratio"], 3),
                    "log_likelihood_ratio": round(row["log_likelihood_ratio"], 3),
                    "time_period": f"{row['window_start'].date()} to {row['window_end'].date()}",
                    "diseases": row["diseases"],
                }
            )
    if anomaly_data is not None:
        top_anomalies = anomaly_data[anomaly_data["is_anomaly"]].head(3)
        for _, row in top_anomalies.iterrows():
            alert_report["top_anomalies"].append(
                {
                    "date": row["date"],
                    "pincode": row["pincode"],
                    "disease": row["disease"],
                    "daily_count": int(row["daily_count"]),
                    "anomaly_score": round(row["anomaly_score"], 3),
                }
            )
    return alert_report


print("\n[5] Generating Alert Report...")
alert = build_alert_report(clusters, anomalies)
print("\nAlert Summary:")
print(f"  Timestamp: {alert['timestamp']}")
print(f"  Clusters: {alert['n_clusters']}")
print(f"  Anomalies: {alert['n_anomalies']}")
print("\nTop Clusters:")
for c in alert["top_clusters"]:
    print(c)
print("\nTop Anomalies:")
for a in alert["top_anomalies"]:
    print(a)
