"""
Loads the trained brain tumor model ONCE at startup.
Custom CBAM layers are redefined here so the model can be deserialized.
"""
import os
import json
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape,
    Multiply, Add, Activation, Concatenate, Conv2D
)


# ---------- Custom CBAM layers (must match training notebook) ----------
@tf.keras.utils.register_keras_serializable(package="cbam")
class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.channels = channels
        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.dense1 = Dense(max(channels // self.ratio, 1), activation='relu', use_bias=False)
        self.dense2 = Dense(channels, activation='linear', use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = self.dense2(self.dense1(self.global_avg_pool(inputs)))
        max_pool = self.dense2(self.dense1(self.global_max_pool(inputs)))
        attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
        attention = Reshape((1, 1, self.channels))(attention)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


@tf.keras.utils.register_keras_serializable(package="cbam")
class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(
            filters=1, kernel_size=self.kernel_size, strides=1, padding='same',
            activation='sigmoid', use_bias=False
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package="cbam")
class CBAM(Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def build(self, input_shape):
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return config


# ---------- Global singleton ----------
_MODEL = None
_CLASS_NAMES = None
_CONFIG = None
_GRAD_MODEL = None


def load_model_once(model_dir: str):
    """Load model, class names, and config once. Reused across requests."""
    global _MODEL, _CLASS_NAMES, _CONFIG, _GRAD_MODEL

    if _MODEL is not None:
        return _MODEL, _CLASS_NAMES, _CONFIG

    model_path = os.path.join(model_dir, "brain_tumor_final.keras")
    class_path = os.path.join(model_dir, "class_names.json")
    config_path = os.path.join(model_dir, "config.json")

    print(f"[model_loader] Loading model from {model_path} ...")
    _MODEL = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
            "CBAM": CBAM,
        },
        compile=False,
    )

    with open(class_path) as f:
        _CLASS_NAMES = json.load(f)
    with open(config_path) as f:
        _CONFIG = json.load(f)

    # Grad-CAM model: outputs [cbam_high feature map, final predictions]
    _GRAD_MODEL = tf.keras.Model(
        inputs=_MODEL.input,
        outputs=[_MODEL.get_layer("cbam_high").output, _MODEL.output],
    )

    print(f"[model_loader] Model loaded. Classes = {_CLASS_NAMES}")
    return _MODEL, _CLASS_NAMES, _CONFIG


def get_grad_model():
    return _GRAD_MODEL
