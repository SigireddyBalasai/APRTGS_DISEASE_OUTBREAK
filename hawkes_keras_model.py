import tensorflow as tf
import numpy as np

class HawkesModel(tf.keras.Model):
    def __init__(self, mu_init=0.5, alpha_init=0.3, beta_init=0.2, gamma_init=0.1, **kwargs):
        super().__init__(**kwargs)
        # Parameters as trainable variables
        self.mu = tf.Variable(mu_init, dtype=tf.float32, trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-6, 10.0), name="mu")
        self.alpha = tf.Variable(alpha_init, dtype=tf.float32, trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-6, 10.0), name="alpha")
        self.beta = tf.Variable(beta_init, dtype=tf.float32, trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-6, 10.0), name="beta")
        self.gamma = tf.Variable(gamma_init, dtype=tf.float32, trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-6, 10.0), name="gamma")

    def call(self, inputs, training=False):
        # Not used for direct prediction, only for fitting
        return self.mu

    def spatial_kernel(self, lat1, lon1, lat2, lon2):
        # Use tf.cond for graph compatibility and always return a tensor
        def ones():
            return tf.ones([1], dtype=tf.float32)
        def kernel():
            R = tf.constant(6371.0, dtype=tf.float32)
            deg2rad = tf.constant(np.pi, dtype=tf.float32) / tf.constant(180.0, dtype=tf.float32)
            lat1_rad = lat1 * deg2rad
            lat2_rad = lat2 * deg2rad
            dlat = (lat2 - lat1) * deg2rad
            dlon = (lon2 - lon1) * deg2rad
            a = (
                tf.math.sin(dlat / 2) ** 2
                + tf.math.cos(lat1_rad) * tf.math.cos(lat2_rad) * tf.math.sin(dlon / 2) ** 2
            )
            c = 2 * tf.math.asin(tf.math.sqrt(a))
            dist = R * c
            return tf.math.exp(-self.gamma * dist)
        result = tf.cond(tf.size(lat2) == 0, ones, kernel)
        # Always return a 1D tensor
        return tf.reshape(result, [-1])


    def intensity(self, t, lat, lon, t_events, lat_events, lon_events):
        # Always flatten event arrays to 1D
        t_events = tf.reshape(t_events, [-1])
        lat_events = tf.reshape(lat_events, [-1])
        lon_events = tf.reshape(lon_events, [-1])
        mask = t_events < t
        past_t = tf.reshape(tf.boolean_mask(t_events, mask), [-1])
        past_lat = tf.reshape(tf.boolean_mask(lat_events, mask), [-1])
        past_lon = tf.reshape(tf.boolean_mask(lon_events, mask), [-1])
        # If any of the past arrays are empty, return mu immediately
        def mu_tensor():
            return tf.convert_to_tensor(self.mu, dtype=tf.float32)
        empty = tf.logical_or(tf.size(past_t) == 0, tf.logical_or(tf.size(past_lat) == 0, tf.size(past_lon) == 0))
        def hawkes():
            time_gaps = t - past_t
            spatial_kernels = self.spatial_kernel(lat, lon, past_lat, past_lon)
            spatial_kernels = tf.reshape(spatial_kernels, [-1])
            hawkes_term = self.alpha * tf.reduce_sum(tf.math.exp(-self.beta * time_gaps) * spatial_kernels)
            return tf.convert_to_tensor(self.mu, dtype=tf.float32) + hawkes_term
        return tf.cond(empty, mu_tensor, hawkes)
    def log_likelihood(self, t_events, lat_events, lon_events, grid_size=20):
        # Always flatten event arrays to 1D
        t_events = tf.reshape(t_events, [-1])
        lat_events = tf.reshape(lat_events, [-1])
        lon_events = tf.reshape(lon_events, [-1])
        # Vectorized log-likelihood with robust empty event handling
        n_ev = t_events.shape[0]
        def compute_ll():
            ll = tf.constant(0.0, dtype=tf.float32)
            for i in range(n_ev):
                t = t_events[i]
                lat = lat_events[i]
                lon = lon_events[i]
                mask = t_events < t
                if tf.reduce_sum(tf.cast(mask, tf.int32)).numpy() == 0:
                    ll += tf.math.log(tf.maximum(self.mu, 1e-10))
                else:
                    intensity_val = self.intensity(t, lat, lon, t_events, lat_events, lon_events)
                    intensity_val = tf.reshape(intensity_val, [])
                    ll += tf.math.log(tf.maximum(intensity_val, 1e-10))

            t_max = tf.reduce_max(t_events)
            time_grid = tf.linspace(0.0, t_max, grid_size)
            dt = t_max / tf.cast(grid_size, tf.float32)
            grid_intensities = []
            for t in time_grid:
                intensities = []
                for idx in range(lat_events.shape[0]):
                    lat_idx = tf.reshape(lat_events[idx], [])
                    lon_idx = tf.reshape(lon_events[idx], [])
                    mask = t_events < t
                    if tf.reduce_sum(tf.cast(mask, tf.int32)).numpy() == 0:
                        intensities.append(self.mu)
                    else:
                        val = self.intensity(t, lat_idx, lon_idx, t_events, lat_events, lon_events)
                        val = tf.reshape(val, [])
                        intensities.append(val)
                intensities = tf.stack(intensities)
                if tf.size(intensities).numpy() > 0:
                    mean_intensity = tf.reduce_mean(intensities)
                else:
                    mean_intensity = tf.convert_to_tensor(self.mu, dtype=tf.float32)
                grid_intensities.append(mean_intensity)
            grid_intensities = tf.stack(grid_intensities)
            integral = tf.reduce_sum(grid_intensities) * dt
            result = ll - integral
            return tf.reshape(result, [])
        if n_ev > 0:
            out = compute_ll()
        else:
            out = tf.constant(0.0, dtype=tf.float32)
        return tf.reshape(out, [])

    def get_config(self):
        return {"mu": float(self.mu.numpy()), "alpha": float(self.alpha.numpy()), "beta": float(self.beta.numpy()), "gamma": float(self.gamma.numpy())}
