
'''
Test 6 different architectures:
1. Classification CNN
2. Classification CNN with ResNet transfer learning
3. Regression CNN
4. Regression CNN with ResNet transfer learning
5. Classification Vision Transformer
6. Regression Vision Transformer'''

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers



def set_callbacks(early_stopping=True, reduce_lr=True, checkpoint=True):
    # Callback to stop model training early if loss stops improving
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_classification_loss',
        patience=5,                # number of epochs to wait for improvement
        restore_best_weights=True, # restore the best weights once training stops
        verbose=1
    )

    # Callback to reduce learning rate if loss stops improving
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_classification_loss',
        factor=0.1,                # the factor by which the learning rate will be reduced
        patience=2,                # number of epochs to wait for improvement
        verbose=1
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True        # save only the best model
    )

    callbacks = []
    if early_stopping:
        callbacks.append(early_stopping_callback)
    if reduce_lr:
        callbacks.append(reduce_lr_callback)
    if checkpoint:
        callbacks.append(checkpoint_callback)

    return callbacks