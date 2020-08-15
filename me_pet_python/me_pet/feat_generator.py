import os
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np


def generate_features(image_root, feature_file, gray=True, base_model=tf.keras.applications.ResNet50V2):
    _, _, file_list = next(os.walk(image_root))
    file_list = [f for f in file_list if f.endswith(".jpg")]

    model = base_model(input_shape=(224, 224, 3), include_top=False)
    model = tf.keras.models.Model(model.input, tf.keras.layers.GlobalAveragePooling2D()(model.output))

    def generator():
        for file in file_list:
            path = f"{image_root}/{file}"
            img = Image.open(path)
            if gray:
                img = img.convert("L")
            img = img.convert("RGB").resize((224, 224))

            img = (np.array(img, dtype=np.float32) / 127.5) - 1

            yield img

    dataset = tf.data.Dataset.from_generator(generator, tf.float32,
                                             tf.TensorShape([224, 224, 3])).batch(32).prefetch(64)

    vectors = model.predict(dataset, use_multiprocessing=True, workers=4, verbose=1)

    feat_vectors = pd.concat([pd.DataFrame(np.array(file_list), columns=["file_name"]),
                              pd.DataFrame(vectors)], axis=1)
    feat_vectors.to_csv(feature_file, index=False)

