import numpy as np
from models import regression_model, LSTM_stateful, tcn, Conv_LSTM
import utils
import tensorflow as tf
keras = tf.keras

def _mean_absolute_deviation(labels, pred):
    mad = tf.abs(pred - tf.reduce_mean(labels)) / len(pred)
    return mad

def conf_score_reg(label, pred,P={'model_type' : 'lstm'}):
    if P['model_type'] == 'lstm':
        model = LSTM_stateful(P)
    elif P['model_type'] == 'tcn':
        model = tcn(P)
    elif P['model_type'] == 'conv_lstm':
        model = Conv_LSTM(P)
    else:
        raise ValueError("model_type must be 'lstm' or 'tcn'")
    
    model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                    loss=[lambda y_true, y_pred: _mean_absolute_deviation(y_true, y_pred)],# [lambda y_true, y_pred: multiple_pinball(y_true, y_pred, self.P['alpha'])], #[lambda y_true, y_pred: _pin_loss(y_true, y_pred, self.P['quantiles'])],
                    run_eagerly=True)
    
    es = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            start_from_epoch=1,
            restore_best_weights=True,
            verbose=1
        )
    
    history = model.fit(label,
                       validation_data=val_data,
                       epochs=epochs,
                       callbacks=[es],
                       verbose=verbose)