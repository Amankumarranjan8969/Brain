"""Grad-CAM computation + colormap overlay."""
import cv2
import numpy as np
import tensorflow as tf
from backend.model_loader import get_grad_model


def compute_gradcam(img_batch: np.ndarray, class_idx: int = None):
    """
    Returns (heatmap, predicted_class_idx).
    img_batch: shape (1, H, W, 3) already preprocessed with densenet.preprocess_input.
    """
    grad_model = get_grad_model()
    if grad_model is None:
        raise RuntimeError("Grad model not initialized. Call load_model_once first.")

    if class_idx is None:
        preds = grad_model.predict(img_batch, verbose=0)[1]
        class_idx = int(np.argmax(preds[0]))

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy(), class_idx


def overlay_heatmap(original_img_uint8: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4, colormap=cv2.COLORMAP_JET):
    """Return RGB image with the heatmap overlaid."""
    h, w = original_img_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
