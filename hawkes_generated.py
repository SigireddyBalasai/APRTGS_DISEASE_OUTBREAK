import tensorflow as tf
import numpy as np


# ===========================================================================
#  Spatiotemporal Hawkes Process  –  Disease Outbreak Model  (Keras-compatible)
# ===========================================================================


class SpatiotemporalHawkes(tf.keras.Model):
    """
    Spatiotemporal Hawkes process for disease-outbreak modelling.

        λ(t, s) = μ  +  α Σ_{tᵢ < t} exp(−β(t − tᵢ)) · exp(−γ · d(s, sᵢ))

    where d(·,·) is the haversine (great-circle) distance in km.

    All four parameters (μ, α, β, γ) are kept positive via a soft-plus
    reparametrisation so that unconstrained optimisers work out of the box.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        mu_init: float = 0.5,
        alpha_init: float = 0.3,
        beta_init: float = 1.0,
        gamma_init: float = 0.1,
        lat_bounds=None,
        lon_bounds=None,
        time_grid_size: int = 50,
        space_grid_size: int = 10,
        num_mixture_components: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store initial values for configuration
        self._mu_init = mu_init
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._gamma_init = gamma_init
        self._lat_bounds = lat_bounds
        self._lon_bounds = lon_bounds
        self._time_grid_size = time_grid_size
        self._space_grid_size = space_grid_size
        self._num_mixture_components = num_mixture_components

        self._log_mu = tf.Variable(
            self._inv_softplus(mu_init),
            dtype=tf.float32,
            trainable=True,
            name="log_mu",
        )
        self._log_alpha = tf.Variable(
            self._inv_softplus(alpha_init),
            dtype=tf.float32,
            trainable=True,
            name="log_alpha",
        )

        # Mixture components for temporal and spatial kernels
        # Initialize mixture weights (will be normalized via softmax)
        self._log_mix_weights_t = tf.Variable(
            tf.zeros([num_mixture_components], dtype=tf.float32),
            trainable=True,
            name="log_mix_weights_t",
        )
        self._log_mix_weights_s = tf.Variable(
            tf.zeros([num_mixture_components], dtype=tf.float32),
            trainable=True,
            name="log_mix_weights_s",
        )

        # Initialize mixture rates for temporal and spatial components (in log space, inverse softplus)
        self._log_betas = tf.Variable(
            tf.ones([num_mixture_components], dtype=tf.float32)
            * self._inv_softplus(beta_init),
            trainable=True,
            name="log_betas",
        )
        self._log_gammas = tf.Variable(
            tf.ones([num_mixture_components], dtype=tf.float32)
            * self._inv_softplus(gamma_init),
            trainable=True,
            name="log_gammas",
        )

    # ---- helpers for the soft-plus constraint -------------------------
    @staticmethod
    def _inv_softplus(x: float) -> float:
        """Inverse of softplus:  log(exp(x) − 1)."""
        if x <= 0:
            raise ValueError(f"Initial value must be > 0, got {x}")
        if x > 20.0:  # avoid overflow in exp
            return float(x)
        return float(np.log(np.exp(x) - 1.0))

    @property
    def mu(self):
        return tf.math.softplus(self._log_mu)

    @property
    def alpha(self):
        return tf.math.softplus(self._log_alpha)

    # Note: For mixture models, beta and gamma properties are not used directly
    # Instead, we use betas and gammas properties which return arrays of mixture components
    @property
    def beta(self):
        # For backward compatibility, return the first beta component
        # In a mixture model, this represents the dominant temporal rate
        betas = self.betas
        return betas[0] if len(betas) > 0 else tf.constant(1.0)

    @property
    def gamma(self):
        # For backward compatibility, return the first gamma component
        # In a mixture model, this represents the dominant spatial rate
        gammas = self.gammas
        return gammas[0] if len(gammas) > 0 else tf.constant(1.0)

    @property
    def branching_ratio(self):
        """α / β  — must be < 1 for stationarity."""
        return self.alpha / self.beta

    @property
    def mix_weights_t(self):
        """Mixture weights for temporal kernel (normalized to sum to 1)."""
        return tf.nn.softmax(self._log_mix_weights_t)

    @property
    def mix_weights_s(self):
        """Mixture weights for spatial kernel (normalized to sum to 1)."""
        return tf.nn.softmax(self._log_mix_weights_s)

    @property
    def betas(self):
        """Mixture rates for temporal kernel."""
        return tf.math.softplus(self._log_betas)

    @property
    def gammas(self):
        """Mixture rates for spatial kernel."""
        return tf.math.softplus(self._log_gammas)

    # ------------------------------------------------------------------
    # haversine utilities
    # ------------------------------------------------------------------
    @staticmethod
    def haversine_distances(lat1, lon1, lat2, lon2):
        """Element-wise haversine distance (km).  Broadcasting-safe."""
        R = 6371.0
        deg2rad = tf.constant(np.pi / 180.0, dtype=tf.float32)

        lat1_r = tf.cast(lat1, tf.float32) * deg2rad
        lat2_r = tf.cast(lat2, tf.float32) * deg2rad
        dlat = (tf.cast(lat2, tf.float32) - tf.cast(lat1, tf.float32)) * deg2rad
        dlon = (tf.cast(lon2, tf.float32) - tf.cast(lon1, tf.float32)) * deg2rad

        a = (
            tf.math.sin(dlat / 2.0) ** 2
            + tf.math.cos(lat1_r) * tf.math.cos(lat2_r) * tf.math.sin(dlon / 2.0) ** 2
        )
        a = tf.clip_by_value(a, 0.0, 1.0)
        return R * 2.0 * tf.math.asin(tf.math.sqrt(a))

    def _pairwise_haversine(self, lat1, lon1, lat2, lon2):
        """Pairwise distance matrix  [N, M]."""
        return self.haversine_distances(
            tf.expand_dims(lat1, 1),
            tf.expand_dims(lon1, 1),
            tf.expand_dims(lat2, 0),
            tf.expand_dims(lon2, 0),
        )

    # ------------------------------------------------------------------
    # mixture kernel computations
    # ------------------------------------------------------------------
    def _temporal_mixture_kernel(self, dt_mat):
        """
        Compute mixture of exponential temporal kernels.

        Args:
            dt_mat: [N, N] matrix of time differences (t_i - t_j)

        Returns:
            [N, N] matrix of temporal kernel values
        """
        # dt_mat should be non-negative (we'll handle negative values in calling code)
        dt_mat = tf.maximum(dt_mat, 0.0)

        # Compute mixture: sum_k w_k * exp(-beta_k * dt)
        # Shape broadcasting: weights [K], betas [K], dt_mat [N,N]
        weights = tf.expand_dims(tf.expand_dims(self.mix_weights_t, 0), 0)  # [1, 1, K]
        betas = tf.expand_dims(tf.expand_dims(self.betas, 0), 0)  # [1, 1, K]
        dt_expanded = tf.expand_dims(dt_mat, -1)  # [N, N, 1]

        # Compute w_k * exp(-beta_k * dt) for each component
        kernel_components = weights * tf.exp(-betas * dt_expanded)  # [N, N, K]

        # Sum over mixture components
        return tf.reduce_sum(kernel_components, axis=-1)  # [N, N]

    def _spatial_mixture_kernel(self, dist_mat):
        """
        Compute mixture of exponential spatial kernels.

        Args:
            dist_mat: [N, N] matrix of spatial distances

        Returns:
            [N, N] matrix of spatial kernel values
        """
        # Compute mixture: sum_k w_k * exp(-gamma_k * dist)
        # Shape broadcasting: weights [K], gammas [K], dist_mat [N,N]
        weights = tf.expand_dims(tf.expand_dims(self.mix_weights_s, 0), 0)  # [1, 1, K]
        gammas = tf.expand_dims(tf.expand_dims(self.gammas, 0), 0)  # [1, 1, K]
        dist_expanded = tf.expand_dims(dist_mat, -1)  # [N, N, 1]

        # Compute w_k * exp(-gamma_k * dist) for each component
        kernel_components = weights * tf.exp(-gammas * dist_expanded)  # [N, N, K]

        # Sum over mixture components
        return tf.reduce_sum(kernel_components, axis=-1)  # [N, N]

    # ------------------------------------------------------------------
    # forward pass  (returns the scalar NLL so Keras can track it)
    # ------------------------------------------------------------------
    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : dict
            't_events'   – [N]   event times (days, hours, …)
            'lat_events' – [N]   event latitudes  (degrees)
            'lon_events' – [N]   event longitudes (degrees)
            'T'          – scalar, end of the observation window

        Returns
        -------
        nll : scalar  negative log-likelihood
        """
        t_ev = tf.cast(tf.reshape(inputs["t_events"], [-1]), tf.float32)
        lat_ev = tf.cast(tf.reshape(inputs["lat_events"], [-1]), tf.float32)
        lon_ev = tf.cast(tf.reshape(inputs["lon_events"], [-1]), tf.float32)
        T = tf.cast(tf.reshape(inputs["T"], []), tf.float32)

        nll = self._neg_log_likelihood(t_ev, lat_ev, lon_ev, T)
        self.add_loss(nll)
        return nll

    # ------------------------------------------------------------------
    # core log-likelihood  (fully vectorised, graph-safe)
    # ------------------------------------------------------------------
    def _neg_log_likelihood(self, t_events, lat_events, lon_events, T):
        mu = self.mu
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        n = tf.shape(t_events)[0]

        # --- sort by time -------------------------------------------------
        idx = tf.argsort(t_events)
        t_sort = tf.gather(t_events, idx)
        la_sort = tf.gather(lat_events, idx)
        lo_sort = tf.gather(lon_events, idx)

        # === TERM 1 :  Σ_i  log λ(tᵢ, sᵢ) ==============================
        #   pair-wise matrices  [N, N]
        dt_mat = (
            tf.expand_dims(t_sort, 1)  # [N,1]
            - tf.expand_dims(t_sort, 0)
        )  # [1,N]  → [N,N]

        #   strict lower-triangular mask  (j < i)
        mask = tf.linalg.band_part(tf.ones([n, n], dtype=tf.float32), -1, 0) - tf.eye(
            n, dtype=tf.float32
        )  # [N,N]

        temporal = self._temporal_mixture_kernel(dt_mat) * mask
        spatial = self._spatial_mixture_kernel(
            self._pairwise_haversine(la_sort, lo_sort, la_sort, lo_sort)
        )  # [N,N]

        trigger_at_events = tf.reduce_sum(temporal * spatial, axis=1)  # [N]
        lam_events = mu + alpha * trigger_at_events  # [N]

        term1 = tf.reduce_sum(tf.math.log(tf.maximum(lam_events, 1e-10)))

        # === TERM 2 :  ∫₀ᵀ ∫_S  λ(t, s) ds dt  (numerical) =============
        la_min, la_max, lo_min, lo_max = self._spatial_bounds(la_sort, lo_sort)

        m_t = self._time_grid_size
        m_s = self._space_grid_size

        time_grid = tf.linspace(0.0, T, m_t)  # [m_t]
        lat_grid = tf.linspace(la_min, la_max, m_s)  # [m_s]
        lon_grid = tf.linspace(lo_min, lo_max, m_s)  # [m_s]

        dt_vol = T / tf.cast(m_t - 1, tf.float32)
        dla_vol = (la_max - la_min) / tf.cast(m_s - 1, tf.float32)
        dlo_vol = (lo_max - lo_min) / tf.cast(m_s - 1, tf.float32)

        # spatial mesh  [m_s², 1]
        la_mesh, lo_mesh = tf.meshgrid(lat_grid, lon_grid, indexing="ij")
        la_flat = tf.reshape(la_mesh, [-1])  # [m_s²]
        lo_flat = tf.reshape(lo_mesh, [-1])  # [m_s²]

        # S[l, i] = exp(−γ d(grid_l , event_i))              [m_s², N]
        S = tf.math.exp(
            -gamma * self._pairwise_haversine(la_flat, lo_flat, la_sort, lo_sort)
        )

        # D[k, i] = Σ_k w_k * exp(−β_k (t_k − tᵢ)) · 1(t_k > tᵢ)      [m_t, N]
        tg = tf.expand_dims(time_grid, 1)  # [m_t, 1]
        te = tf.expand_dims(t_sort, 0)  # [1,  N]
        dt_grid = tg - te  # [m_t, N]
        # Compute mixture temporal kernel for integral
        D = self._temporal_mixture_kernel(dt_grid) * tf.cast(dt_grid > 0.0, tf.float32)

        # λ_grid[k, l] = μ + α · Σ_i D[k,i] S[l,i]
        #              = μ + α · (D  @  Sᵀ)[k, l]            [m_t, m_s²]
        trigger_grid = tf.matmul(D, tf.transpose(S))
        lam_grid = mu + alpha * trigger_grid

        integral = tf.reduce_sum(lam_grid) * dt_vol * dla_vol * dlo_vol

        return -(term1 - integral)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # inference helpers
    # ------------------------------------------------------------------
    def intensity_at_points(
        self, t_query, lat_query, lon_query, t_events, lat_events, lon_events
    ):
        """
        Compute λ(t, s) at arbitrary query points given an event history.

        Returns
        -------
        intensities : [Q]
        """
        tq = tf.cast(tf.reshape(t_query, [-1]), tf.float32)
        laq = tf.cast(tf.reshape(lat_query, [-1]), tf.float32)
        loq = tf.cast(tf.reshape(lon_query, [-1]), tf.float32)
        te = tf.cast(tf.reshape(t_events, [-1]), tf.float32)
        lae = tf.cast(tf.reshape(lat_events, [-1]), tf.float32)
        loe = tf.cast(tf.reshape(lon_events, [-1]), tf.float32)

        dt = tf.expand_dims(tq, 1) - tf.expand_dims(te, 0)  # [Q, N]
        D = self._temporal_mixture_kernel(dt) * tf.cast(dt > 0.0, tf.float32)

        S = self._spatial_mixture_kernel(self._pairwise_haversine(laq, loq, lae, loe))

        return self.mu + self.alpha * tf.reduce_sum(D * S, axis=1)

    def predict_risk_map(
        self, t_query, lat_grid, lon_grid, t_events, lat_events, lon_events
    ):
        """
        Spatial risk map at one point in time.

        Returns
        -------
        risk : [len(lat_grid), len(lon_grid)]
        """
        la_m, lo_m = tf.meshgrid(lat_grid, lon_grid, indexing="ij")
        la_f = tf.reshape(la_m, [-1])
        lo_f = tf.reshape(lo_m, [-1])
        t_f = tf.fill(tf.shape(la_f), tf.cast(t_query, tf.float32))

        intensities = self.intensity_at_points(
            t_f, la_f, lo_f, t_events, lat_events, lon_events
        )
        return tf.reshape(intensities, tf.shape(la_m))

    # ------------------------------------------------------------------
    # serialisation (Keras save / load)
    # ------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            mu_init=self._mu_init,
            alpha_init=self._alpha_init,
            beta_init=self._beta_init,
            gamma_init=self._gamma_init,
            lat_bounds=self._lat_bounds,
            lon_bounds=self._lon_bounds,
            time_grid_size=self._time_grid_size,
            space_grid_size=self._space_grid_size,
            num_mixture_components=self._num_mixture_components,
        )
        return cfg

    @classmethod
    def from_config(cls, config):
        # Extract mixture components parameter if present
        num_mixture_components = config.pop("num_mixture_components", 2)
        return cls(num_mixture_components=num_mixture_components, **config)


# ===========================================================================
#  Custom training loop that talks to Keras optimisers
# ===========================================================================
class HawkesTrainer:
    """Thin wrapper around a ``SpatiotemporalHawkes`` for MLE fitting."""

    def __init__(self, model: SpatiotemporalHawkes, optimizer=None):
        self.model = model
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_history: list[float] = []

    @tf.function
    def _train_step(self, t_ev, la_ev, lo_ev, T):
        with tf.GradientTape() as tape:
            nll = self.model(
                {"t_events": t_ev, "lat_events": la_ev, "lon_events": lo_ev, "T": T},
                training=True,
            )
        grads = tape.gradient(nll, self.model.trainable_variables)
        grads = [
            tf.clip_by_value(g, -10.0, 10.0) if g is not None else g for g in grads
        ]
        self.optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(grads, self.model.trainable_variables)
                if g is not None
            ]
        )
        return nll

    def fit(
        self,
        t_events,
        lat_events,
        lon_events,
        T,
        epochs: int = 200,
        verbose: bool = True,
    ):
        t_ev = tf.constant(t_events, dtype=tf.float32)
        la_ev = tf.constant(lat_events, dtype=tf.float32)
        lo_ev = tf.constant(lon_events, dtype=tf.float32)
        T_c = tf.constant(T, dtype=tf.float32)

        for epoch in range(epochs):
            nll = float(self._train_step(t_ev, la_ev, lo_ev, T_c).numpy())
            self.loss_history.append(nll)

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                m = self.model
                print(
                    f"Epoch {epoch:4d} │ NLL {nll:+12.2f} │ "
                    f"μ={float(m.mu.numpy()):.4f}  "
                    f"α={float(m.alpha.numpy()):.4f}  "
                    f"β={float(m.beta.numpy()):.4f}  "
                    f"γ={float(m.gamma.numpy()):.4f}  "
                    f"ratio={float(m.branching_ratio.numpy()):.4f}"
                )
        return self.loss_history


# ===========================================================================
#  Synthetic data generator  (Ogata thinning)
# ===========================================================================
def simulate_hawkes_spatiotemporal(
    mu=0.5,
    alpha=0.3,
    beta=1.0,
    gamma=0.05,
    T=100.0,
    lat_center=40.7,
    lon_center=-74.0,
    spatial_std=0.3,
    seed=42,
):
    """Generate a spatiotemporal Hawkes realisation via Ogata thinning."""
    rng = np.random.default_rng(seed)

    ev_t, ev_lat, ev_lon = [], [], []
    t = 0.0
    lam_star = mu * 5.0

    while t < T:
        t += rng.exponential(1.0 / lam_star)
        if t >= T:
            break

        lam = mu
        for j in range(len(ev_t)):
            dt = t - ev_t[j]
            dlat = np.radians(lat_center - ev_lat[j])
            dlon = np.radians(lon_center - ev_lon[j])
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(ev_lat[j]))
                * np.cos(np.radians(lat_center))
                * np.sin(dlon / 2) ** 2
            )
            dist_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(min(a, 1.0)))
            lam += alpha * np.exp(-beta * dt) * np.exp(-gamma * dist_km)

        if lam_star < lam:
            lam_star = lam * 2.0

        if rng.uniform() < lam / lam_star:
            ev_t.append(t)
            # new event location: cluster around triggering event or random
            if len(ev_t) > 1 and rng.uniform() < alpha / (mu + alpha):
                # triggered event – cluster around a random parent
                parent = rng.integers(0, len(ev_t) - 1)
                ev_lat.append(ev_lat[parent] + rng.normal(0, spatial_std))
                ev_lon.append(ev_lon[parent] + rng.normal(0, spatial_std))
            else:
                # background event
                ev_lat.append(lat_center + rng.normal(0, spatial_std))
                ev_lon.append(lon_center + rng.normal(0, spatial_std))

        lam_star = lam * 1.5 + mu

    return (
        np.array(ev_t, dtype=np.float32),
        np.array(ev_lat, dtype=np.float32),
        np.array(ev_lon, dtype=np.float32),
    )


# ===========================================================================
#  Evaluation metrics
# ===========================================================================
class HawkesEvaluator:
    """Utilities for evaluating a fitted spatiotemporal Hawkes model."""

    def __init__(self, model: SpatiotemporalHawkes):
        self.model = model

    def residual_analysis(self, t_events, lat_events, lon_events):
        """
        Compute transformed times (rescaled residuals) for GoF testing.
        Under correct model, Λ(tᵢ) − Λ(tᵢ₋₁) ~ Exp(1).
        """
        t_ev = tf.cast(tf.reshape(t_events, [-1]), tf.float32)
        la_ev = tf.cast(tf.reshape(lat_events, [-1]), tf.float32)
        lo_ev = tf.cast(tf.reshape(lon_events, [-1]), tf.float32)

        idx = tf.argsort(t_ev)
        t_sort = tf.gather(t_ev, idx)
        la_sort = tf.gather(la_ev, idx)
        lo_sort = tf.gather(lo_ev, idx)

        n = t_sort.shape[0]
        compensators = []
        for i in range(1, n):
            # numerical integral of λ between consecutive events
            n_grid = 20
            t_lo = t_sort[i - 1]
            t_hi = t_sort[i]
            t_grid = tf.linspace(t_lo, t_hi, n_grid)
            dt = (t_hi - t_lo) / tf.cast(n_grid - 1, tf.float32)

            vals = self.model.intensity_at_points(
                t_grid,
                tf.fill([n_grid], la_sort[i]),
                tf.fill([n_grid], lo_sort[i]),
                t_ev,
                la_ev,
                lo_ev,
            )
            compensators.append(float(tf.reduce_sum(vals).numpy() * dt))

        return np.array(compensators, dtype=np.float32)

    def aic(self, t_events, lat_events, lon_events, T):
        """Akaike Information Criterion:  2k − 2 log L."""
        nll = float(
            self.model(
                {
                    "t_events": t_events,
                    "lat_events": lat_events,
                    "lon_events": lon_events,
                    "T": T,
                },
                training=False,
            ).numpy()
        )
        k = len(self.model.trainable_variables)
        return 2.0 * k + 2.0 * nll

    def bic(self, t_events, lat_events, lon_events, T):
        """Bayesian Information Criterion:  k ln(n) − 2 log L."""
        n = len(t_events)
        nll = float(
            self.model(
                {
                    "t_events": t_events,
                    "lat_events": lat_events,
                    "lon_events": lon_events,
                    "T": T,
                },
                training=False,
            ).numpy()
        )
        k = len(self.model.trainable_variables)
        return k * np.log(n) + 2.0 * nll


# ===========================================================================
#  Keras callback for logging Hawkes-specific metrics
# ===========================================================================
class HawkesCallback(tf.keras.callbacks.Callback):
    """Custom Keras callback that prints Hawkes parameters every N epochs."""

    def __init__(self, print_every: int = 10):
        super().__init__()
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every == 0:
            m = self.model
            print(
                f"  [Hawkes] μ={float(m.mu.numpy()):.4f}  "
                f"α={float(m.alpha.numpy()):.4f}  "
                f"β={float(m.beta.numpy()):.4f}  "
                f"γ={float(m.gamma.numpy()):.4f}  "
                f"branching_ratio={float(m.branching_ratio.numpy()):.4f}"
            )


# ===========================================================================
#  Keras-style dataset wrapper
# ===========================================================================
class HawkesDataset:
    """Wraps event data into the dict format expected by the model."""

    def __init__(self, t_events, lat_events, lon_events, T):
        self.t_events = np.asarray(t_events, dtype=np.float32)
        self.lat_events = np.asarray(lat_events, dtype=np.float32)
        self.lon_events = np.asarray(lon_events, dtype=np.float32)
        self.T = float(T)

    def as_dict(self):
        return {
            "t_events": tf.constant(self.t_events),
            "lat_events": tf.constant(self.lat_events),
            "lon_events": tf.constant(self.lon_events),
            "T": tf.constant(self.T),
        }

    def summary(self):
        print(f"Events        : {len(self.t_events)}")
        print(f"Time range    : [0, {self.T:.2f}]")
        print(
            f"Lat range     : [{self.lat_events.min():.4f}, "
            f"{self.lat_events.max():.4f}]"
        )
        print(
            f"Lon range     : [{self.lon_events.min():.4f}, "
            f"{self.lon_events.max():.4f}]"
        )


# ===========================================================================
#  Full usage example
# ===========================================================================
def main():
    print("=" * 70)
    print("  Spatiotemporal Hawkes Process – Disease Outbreak Modelling")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Simulate synthetic disease outbreak data
    # ------------------------------------------------------------------
    TRUE_MU = 0.5
    TRUE_ALPHA = 0.3
    TRUE_BETA = 1.0
    TRUE_GAMMA = 0.05
    T_MAX = 50.0

    print("\n[1] Simulating synthetic data …")
    t_events, lat_events, lon_events = simulate_hawkes_spatiotemporal(
        mu=TRUE_MU,
        alpha=TRUE_ALPHA,
        beta=TRUE_BETA,
        gamma=TRUE_GAMMA,
        T=T_MAX,
        lat_center=40.7,
        lon_center=-74.0,
        spatial_std=0.3,
        seed=42,
    )

    dataset = HawkesDataset(t_events, lat_events, lon_events, T=T_MAX)
    dataset.summary()

    print(
        f"\nTrue parameters: μ={TRUE_MU}, α={TRUE_ALPHA}, β={TRUE_BETA}, γ={TRUE_GAMMA}"
    )

    # ------------------------------------------------------------------
    # 2) Build and inspect model
    # ------------------------------------------------------------------
    print("\n[2] Building SpatiotemporalHawkes model …")
    model = SpatiotemporalHawkes(
        mu_init=0.8,
        alpha_init=0.5,
        beta_init=0.5,
        gamma_init=0.2,
        lat_bounds=(39.5, 42.0),
        lon_bounds=(-75.5, -72.5),
        time_grid_size=40,
        space_grid_size=8,
    )

    # Dry-run so Keras builds the graph
    _ = model(dataset.as_dict(), training=False)
    model.summary()

    # ------------------------------------------------------------------
    # 3) Fit using custom training loop
    # ------------------------------------------------------------------
    print("\n[3] Fitting via MLE (Adam) …")
    trainer = HawkesTrainer(
        model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    loss_history = trainer.fit(
        t_events,
        lat_events,
        lon_events,
        T=T_MAX,
        epochs=300,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 4) Print recovered parameters
    # ------------------------------------------------------------------
    print("\n[4] Recovered parameters:")
    print(f"    μ = {float(model.mu.numpy()):.4f}   (true {TRUE_MU})")
    print(f"    α = {float(model.alpha.numpy()):.4f}   (true {TRUE_ALPHA})")
    print(f"    β = {float(model.beta.numpy()):.4f}   (true {TRUE_BETA})")
    print(f"    γ = {float(model.gamma.numpy()):.4f}   (true {TRUE_GAMMA})")
    print(
        f"    branching ratio = "
        f"{float(model.branching_ratio.numpy()):.4f}   "
        f"(true {TRUE_ALPHA / TRUE_BETA:.4f})"
    )

    # ------------------------------------------------------------------
    # 5) Model evaluation
    # ------------------------------------------------------------------
    print("\n[5] Model evaluation:")
    evaluator = HawkesEvaluator(model)
    aic = evaluator.aic(t_events, lat_events, lon_events, T_MAX)
    bic = evaluator.bic(t_events, lat_events, lon_events, T_MAX)
    print(f"    AIC = {aic:.2f}")
    print(f"    BIC = {bic:.2f}")

    # ------------------------------------------------------------------
    # 6) Risk map at the last observed time
    # ------------------------------------------------------------------
    print("\n[6] Generating risk map …")
    lat_grid = tf.linspace(39.5, 42.0, 30)
    lon_grid = tf.linspace(-75.5, -72.5, 30)
    t_query = float(t_events.max())

    risk_map = model.predict_risk_map(
        t_query,
        lat_grid,
        lon_grid,
        t_events,
        lat_events,
        lon_events,
    )
    print(f"    Risk map shape : {risk_map.shape}")
    print(f"    Min intensity  : {float(tf.reduce_min(risk_map).numpy()):.4f}")
    print(f"    Max intensity  : {float(tf.reduce_max(risk_map).numpy()):.4f}")
    print(f"    Mean intensity : {float(tf.reduce_mean(risk_map).numpy()):.4f}")

    # ------------------------------------------------------------------
    # 7) Save & reload (Keras serialization)
    # ------------------------------------------------------------------
    print("\n[7] Save / load test …")
    config = model.get_config()
    print(f"    Config: {config}")
    model_reloaded = SpatiotemporalHawkes.from_config(config)
    _ = model_reloaded(dataset.as_dict(), training=False)
    nll_orig = float(model(dataset.as_dict()).numpy())
    nll_reload = float(model_reloaded(dataset.as_dict()).numpy())
    print(f"    NLL original : {nll_orig:.2f}")
    print(f"    NLL reloaded : {nll_reload:.2f}")
    print(f"    Match: {np.isclose(nll_orig, nll_reload, atol=1e-3)}")

    # ------------------------------------------------------------------
    # 8) Optional: plot loss curve if matplotlib is available
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        axes[0].plot(loss_history)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Negative Log-Likelihood")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)

        # Risk map
        rm_np = risk_map.numpy()
        im = axes[1].imshow(
            rm_np,
            origin="lower",
            extent=[-75.5, -72.5, 39.5, 42.0],
            aspect="auto",
            cmap="hot",
        )
        axes[1].scatter(
            lon_events,
            lat_events,
            c="cyan",
            s=8,
            alpha=0.6,
            label="events",
        )
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        axes[1].set_title(f"Risk Map at t={t_query:.1f}")
        axes[1].legend()
        plt.colorbar(im, ax=axes[1], label="λ(t,s)")
        plt.tight_layout()
        plt.savefig("hawkes_results.png", dpi=150)
        plt.show()
        print("\n    Plot saved to hawkes_results.png")
    except ImportError:
        print("\n    matplotlib not available – skipping plots.")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
