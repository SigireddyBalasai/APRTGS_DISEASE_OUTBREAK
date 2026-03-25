import tensorflow as tf
import numpy as np
from hawkes_keras_model import HawkesModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

# Helper to create tf.data.Dataset for Hawkes log-likelihood
class HawkesDataset(tf.data.Dataset):
    def __new__(cls, events_np):
        t = events_np[:, 0].astype(np.float32)
        lat = events_np[:, 1].astype(np.float32)
        lon = events_np[:, 2].astype(np.float32)
        # Each sample is (t, lat, lon)
        return tf.data.Dataset.from_tensor_slices((t, lat, lon))

# Custom loss for Hawkes log-likelihood
class HawkesLoss(tf.keras.losses.Loss):
    def __init__(self, t_events, lat_events, lon_events, grid_size=20):
        super().__init__()
        self.t_events = tf.constant(t_events, dtype=tf.float32)
        self.lat_events = tf.constant(lat_events, dtype=tf.float32)
        self.lon_events = tf.constant(lon_events, dtype=tf.float32)
        self.grid_size = grid_size

    def call(self, y_true, y_pred):
        # y_true, y_pred are not used; loss is computed from model state
        model = self.model
        return -model.log_likelihood(self.t_events, self.lat_events, self.lon_events, grid_size=self.grid_size)

# Main training function

def train_hawkes_model(events_np, epochs=100, val_split=0.2, lr=0.05, batch_size=None):
    # Time-based split
    n_events = len(events_np)
    split_idx = int(n_events * (1 - val_split))
    events_np = events_np[events_np[:, 0].argsort()]
    train_np = events_np[:split_idx]
    val_np = events_np[split_idx:]

    # Prepare datasets (no batching, as loss is global)
    train_ds = HawkesDataset(train_np).batch(len(train_np))
    val_ds = HawkesDataset(val_np).batch(len(val_np))

    # Debug: print the shape of the batch
    for batch in train_ds.take(1):
        print("Train batch shapes:", [x.shape for x in batch])
    for batch in val_ds.take(1):
        print("Val batch shapes:", [x.shape for x in batch])

    # Model
    model = HawkesModel()
    # Attach loss to model for callbacks
    # Allow grid_size to be tuned for speed/accuracy tradeoff
    grid_size = 10  # Lower for speed, higher for accuracy
    loss_fn = HawkesLoss(train_np[:,0], train_np[:,1], train_np[:,2], grid_size=grid_size)
    loss_fn.model = model

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Checkpoints, logging, early stopping
    checkpoint_cb = ModelCheckpoint('hawkes_checkpoint.keras', save_best_only=True, monitor='val_loss', mode='min')
    earlystop_cb = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', mode='min')
    csvlogger_cb = CSVLogger('hawkes_training_log.csv')
    callbacks = [checkpoint_cb, earlystop_cb, csvlogger_cb]

    # Compile
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Fit
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Hawkes Process Training/Validation Loss (Keras)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, history
