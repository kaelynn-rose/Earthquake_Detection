
import sys
import time

sys.path.append('../')

import tensorflow as tf


def callbacks_setup(model_tag, epochs):
    # Callback to stop model training early if loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,                # number of epochs to wait for improvement
        restore_best_weights=True, # restore the best weights once training stops
        verbose=1
    )

    # Callback to reduce learning rate if loss stops improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,                # the factor by which the learning rate will be reduced
        patience=3,                # number of epochs to wait for improvement
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'../models/{model_tag}_{epochs}epochs_{int(time.time())}.keras',
        monitor='val_loss',
        save_best_only=True        # save only the best model
    )
    return [early_stopping, reduce_lr, checkpoint]


def image_preprocessing(image, image_size):
    image = tf.image.resize(image, image_size)  # Resize image
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image


def build_compile_classification_cnn(
    learning_rate=1e-6, loss='binary_crossentropy', metrics=['accuracy']
):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def build_compile_regression_cnn(
    learning_rate=1e-5, loss='mse', metrics=['mae']
):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation = 'relu', padding = 'same'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def build_compile_classification_lstm(
    input_shape, learning_rate=1e-5, loss='binary_crossentropy', metrics=['accuracy']
):
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def build_compile_regression_lstm(
    input_shape, learning_rate=1e-4, loss='mse', metrics=['mae']
):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model