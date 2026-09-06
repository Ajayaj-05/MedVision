import tensorflow as tf
import numpy as np


def make_gradcam_heatmap(image, model, base_model):

    target_layer = base_model.get_layer("relu")

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=target_layer.output
    )

    classifier = model.layers[-1]

    rgb_image = tf.keras.layers.Concatenate(axis=-1)(
        [image, image, image]
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_model(
            rgb_image,
            training=False
        )

        pooled_features = tf.reduce_mean(
            conv_outputs,
            axis=(1, 2)
        )

        predictions = classifier(pooled_features)

        pneumonia_score = predictions[:, 0]

    gradients = tape.gradient(
        pneumonia_score,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()