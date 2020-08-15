import tensorflow as tf
import pandas as pd
import numpy as np


def get_base_model(feature_file="dog_feats.csv", mp_size=8, base_model=tf.keras.applications.ResNet50V2):
    """
    Args:
        feature_file: feature file generated from base_model
        mp_size: max pool size
        base_model: base model to extract image features

    Returns:
        model: tensorflow keras model
        f_names: file name list
    """
    feats = pd.read_csv(feature_file)
    f_names = feats.values[:, 0]
    feats = feats.values[:, 1:].astype(np.float32)
    feats = tf.nn.max_pool1d(tf.expand_dims(feats, axis=-1), int(mp_size*1.5), mp_size, padding='SAME')
    feats = tf.reshape(feats, [-1, int(2048//mp_size)])

    model = base_model(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    feats_out = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    feats_out = tf.nn.max_pool1d(tf.expand_dims(feats_out, axis=-1), int(mp_size*1.5), mp_size, padding='SAME')
    feats_out = tf.reshape(feats_out, [-1, int(2048//mp_size)])

    out_layer = tf.expand_dims(feats_out, axis=1) - feats
    out_layer = tf.square(out_layer)
    out_layer = tf.reduce_sum(out_layer, axis=-1)
    out_idx = tf.argsort(out_layer)

    out_layer = (out_layer - tf.math.reduce_mean(out_layer, axis=-1, keepdims=True)) / tf.math.reduce_std(out_layer, axis=-1, keepdims=True)
    out_layer = out_layer - tf.reduce_min(out_layer, axis=-1, keepdims=True)
    out_layer = 1 - (out_layer / tf.reduce_max(out_layer, axis=-1, keepdims=True))

    model = tf.keras.models.Model(model.input, [out_idx, out_layer])

    return model, f_names
