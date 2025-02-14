"""Utilities for ML model training and evaluation."""

import sys

sys.path.append('../')
sys.path.append('../../../')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy.stats import mode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def prepare_datasets(
    imgs, labels, preproc_func=None, preproc_func_kwargs=None,
    use_scaler=False, batch_size=32
):
    # If array has only one dimension, it must be reshaped for input into the scaler
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, random_state=0, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

    if use_scaler:
        scaler = RobustScaler()
        y_train = scaler.fit_transform(y_train)
        y_val = scaler.transform(y_val)
        y_test = scaler.transform(y_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    if preproc_func:
        train_dataset = train_dataset.map(lambda x, y: (preproc_func(x, **preproc_func_kwargs), y))
        val_dataset = val_dataset.map(lambda x, y: (preproc_func(x, **preproc_func_kwargs), y))
        test_dataset = test_dataset.map(lambda x, y: (preproc_func(x, **preproc_func_kwargs), y))

    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if use_scaler:
        return train_dataset, val_dataset, test_dataset, scaler
    else:
        return train_dataset, val_dataset, test_dataset


def plot_training_history(history):
    if 'accuracy' and 'val_accuracy' in history.history.keys():
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.plot(history.history['loss'], color='xkcd:cranberry', label='Training loss')
        ax1.plot(history.history['val_loss'], color='xkcd:dusty blue', label='Validation loss')
        ax1.grid(True, alpha=0.2, zorder=5)
        ax1.set_title('Model loss')
        ax1.set_ylabel('Loss (binary crossentropy)')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax2.plot(history.history['accuracy'], color='xkcd:cranberry', label='Training accuracy')
        ax2.plot(history.history['val_accuracy'], color='xkcd:dusty blue', label='Validation accuracy')
        ax2.grid(True, alpha=0.2, zorder=5)
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        plt.tight_layout()
        plt.show()
    elif 'mae' and 'val_mae' in history.history.keys():
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.plot(history.history['loss'], color='xkcd:cranberry', label='Training loss')
        ax1.plot(history.history['val_loss'], color='xkcd:dusty blue', label='Validation loss')
        ax1.grid(True, alpha=0.2, zorder=5)
        ax1.set_title('Model loss')
        ax1.set_ylabel('Loss (Mean Squared Error)')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax2.plot(history.history['mae'], color='xkcd:cranberry', label='Training MAE')
        ax2.plot(history.history['val_mae'], color='xkcd:dusty blue', label='Validation MAE')
        ax2.grid(True, alpha=0.2, zorder=5)
        ax2.set_title('Mean Absolute Error (MAE)')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        plt.tight_layout()
        plt.show()
    elif 'loss' and 'val_loss' in history.history.keys():
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(history.history['loss'], color='xkcd:cranberry', label='Training loss')
        ax.plot(history.history['val_loss'], color='xkcd:dusty blue', label='Validation loss')
        ax.grid(True, alpha=0.2, zorder=5)
        ax.set_title('Model loss')
        ax.set_ylabel('Loss (Mean Squared Error)')
        ax.set_xlabel('Epoch')
        ax.legend()
        plt.tight_layout()
        plt.show()


def evaluate_classification_model(trained_model, train_dataset, test_dataset):
    # Fetch labels
    train_labels = []
    for images, labels in train_dataset:
        train_labels.append(np.array(labels))
    train_labels = np.concatenate(train_labels)

    test_labels = []
    for _, labels in test_dataset:
        test_labels.append(labels.numpy())
    test_labels = np.concatenate(test_labels)

    # Get baseline evaluation metrics
    most_common_label = mode(train_labels[:,0])[0]
    predicted_classes_baseline = np.full((len(test_labels)), most_common_label, dtype=int)

    accuracy_base = accuracy_score(test_labels, predicted_classes_baseline)
    precision_base = precision_score(test_labels, predicted_classes_baseline)
    recall_base = recall_score(test_labels, predicted_classes_baseline)
    f1_base = f1_score(test_labels, predicted_classes_baseline)
    baseline_metrics = [accuracy_base, precision_base, recall_base, f1_base]
    print(f'Model baseline accuracy: {accuracy_base}\n Model baseline precision: {precision_base}\n Model baseline recall: {recall_base}\n Model baseline F1 score: {f1_base}')

    # predict the class of each image
    pred_probs = trained_model.predict(test_dataset)
    predicted_classes = (pred_probs > 0.5).astype(int)

    # Get test dataset evaluation metrics
    accuracy = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes)
    recall = recall_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes)
    metrics = [accuracy, precision, recall, f1]
    print(f'Model accuracy: {accuracy}\n Model precision: {precision}\n Model recall: {recall}\n Model F1 score: {f1}')

    # Create confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes) # compare target values to predicted values and show confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not earthquake','earthquake'])

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(test_labels, predicted_classes)
    roc_auc = auc(fpr, tpr)

    # Plot accuracy history, confusion matrix, roc curve
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,6))
    disp.plot(cmap='Blues', ax=ax1, values_format='')
    ax1.set_title('Test Dataset Confusion Matrix')
    ax2_x = np.arange(0, len(metrics))
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    bar_container1 = ax2.bar(ax2_x-0.2, baseline_metrics, width=0.39, color='xkcd:cranberry', zorder=5, label='Baseline model metrics')
    bar_container2 = ax2.bar(ax2_x+0.2, metrics, width=0.39, color='xkcd:french blue', zorder=5, label='Model metrics')
    ax2.bar_label(bar_container1, fmt='{:.4f}')
    ax2.bar_label(bar_container2, fmt='{:.4f}')
    ax2.set_xticks(ax2_x)
    ax2.set_xticklabels(metrics_labels)
    ax2.set_ylim([0,1.25])
    ax2.set_ylabel('Metric value')
    ax2.set_title('Test Dataset Metrics')
    ax2.grid(True, alpha=0.2, zorder=0)
    ax2.legend()
    ax3.plot(fpr, tpr, color='xkcd:cranberry', label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line (random classifier)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Test Dataset ROC Curve')
    ax3.grid(True, alpha=0.2, zorder=0)
    plt.tight_layout()
    plt.show()


def evaluate_regression_model(trained_model, train_dataset, test_dataset, target_variable_name, scaler=None):
    # Fetch labels
    train_labels = []
    for images, labels in train_dataset:
        train_labels.append(np.array(labels))
    train_labels = np.concatenate(train_labels)

    test_labels = []
    for images, labels in test_dataset:
        test_labels.append(np.array(labels))
    test_labels = np.concatenate(test_labels)

    # Calculate baseline error
    baseline_residuals = test_labels - np.mean(train_labels)
    baseline_mse = np.mean(baseline_residuals**2)
    baseline_mae = np.mean(np.abs(baseline_residuals))

    # Evaluate model
    print('Evaluating model on test dataset')
    test_results = trained_model.evaluate(test_dataset, verbose=1)
    print(f'Test data MSE: {test_results[0]}')
    print(f'Test data MAE: {test_results[1]}')

    print('Getting predictions')
    predicted = trained_model.predict(test_dataset)

    if scaler:
        test_labels = scaler.inverse_transform(test_labels)
        predicted = scaler.inverse_transform(predicted)
        print(predicted)

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,6), gridspec_kw={'width_ratios':[1,1,2]})
    baseline_metrics = [baseline_mse, baseline_mae]
    model_metrics = [test_results[0], test_results[1]]
    bar_container1 = ax1.bar(1-0.2, baseline_metrics[0], color='xkcd:cranberry', width=0.35, label='Baseline')
    bar_container2 = ax1.bar(1+0.2, model_metrics[0], color='xkcd:french blue', width=0.35, label='Model')
    ax1.bar_label(bar_container1, fmt='{:.2f}')
    ax1.bar_label(bar_container2, fmt='{:.2f}')
    ax1.set_xlim([0.4,1.6])
    ax1.set_xticks([1-0.2, 1+0.2])
    ax1.set_xticklabels(['Baseline\nmodel','Model'])
    ax1.set_title('Model vs. baseline MSE')
    ax1.set_ylabel('Mean squared error (MSE)')
    bar_container3 = ax2.bar(1-0.2, baseline_metrics[1], color='xkcd:cranberry', width=0.35, label='Baseline')
    bar_container4 = ax2.bar(1+0.2, model_metrics[1], color='xkcd:french blue', width=0.35, label='Model')
    ax2.bar_label(bar_container3, fmt='{:.2f}')
    ax2.bar_label(bar_container4, fmt='{:.2f}')
    ax2.set_xlim([0.4,1.6])
    ax2.set_xticks([1-0.2, 1+0.2])
    ax2.set_xticklabels(['Baseline\nmodel','Model'])
    ax2.set_ylabel('Mean absolute error (MAE)')
    ax2.set_title('Model vs. baseline MAE')
    plot_max = max(test_labels.max(), predicted.max())
    plot_min = min(-0.5, test_labels.min(), predicted.min())
    ax3.scatter(test_labels, predicted, color='xkcd:french blue', alpha=0.05)
    ax3.plot([plot_min*0.9,plot_max*1.1], [plot_min*0.9,plot_max*1.1], color='black', linestyle='--')
    ax3.set_ylabel('Predicted value')
    ax3.set_xlabel('Observed value')
    ax3.set_title(f'Model results: {target_variable_name}')
    ax3.set_xlim([plot_min*0.9, plot_max*1.1])
    ax3.set_ylim([plot_min*0.9, plot_max*1.1])
    ax3.grid(True, alpha=0.2, zorder=5)
    plt.tight_layout()
    plt.show()