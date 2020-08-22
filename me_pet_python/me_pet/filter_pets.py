from models import get_base_model
import tensorflow as tf
import numpy as np
import os
from PIL import Image


def generate_filter_list_faces(face_root, feature_file, filter_file, n_top=1000, threshold=0.5, mp_size=8,
                               base_model=tf.keras.applications.ResNet50V2):
    _, _, file_list = next(os.walk(face_root))
    file_list = [f for f in file_list if f.endswith(".jpg") or f.endswith(".png")]
    model, f_names = get_base_model(feature_file=feature_file, mp_size=mp_size, base_model=base_model)
    images = [Image.open(f"{face_root}/{file}") for file in file_list]
    images += [image.convert("L") for image in images]
    images = [image.convert("RGB").resize((224, 224)) for image in images]
    images = np.array([(np.array(image, dtype=np.float32)/127.5)-1 for image in images], dtype=np.float32)

    close_idx, feat_diff = model.predict(images, batch_size=64, verbose=1)
    if os.path.isfile(filter_file):
        with open(filter_file, 'r') as f:
            filter_list = f.read().split("\n")
        filter_list = [name for name in filter_list if name != ""]
    else:
        filter_list = []

    n_test = len(images)

    idx_histogram = np.zeros((len(f_names),), dtype=np.int)
    for i in range(n_test):
        idx_histogram[close_idx[i, :n_top]] += 1

    idx_histogram = idx_histogram / n_test

    filter_idx = np.argwhere(idx_histogram > threshold).flatten()
    filter_list += f_names[filter_idx].tolist()
    filter_list = list(set(filter_list))

    with open(filter_file, "w") as f:
        for file_name in filter_list:
            f.write(f"{file_name}\n")


def generate_filter_list_random(feature_file, filter_file, n_top=1000, n_test=3000, threshold=0.5, mp_size=8,
                                base_model=tf.keras.applications.ResNet50V2):
    model, f_names = get_base_model(feature_file=feature_file, mp_size=mp_size, base_model=base_model)

    dummy_gray1 = tf.random.uniform([n_test, 224, 224], minval=-1.0, maxval=1.0)
    dummy_gray2 = tf.random.uniform([n_test, 224, 224], minval=-1.0, maxval=0.0)
    dummy_gray3 = tf.random.uniform([n_test, 224, 224], minval=0.0, maxval=1.0)
    dummy_gray = tf.concat([dummy_gray1, dummy_gray2, dummy_gray3], axis=0)
    dummy_gray = tf.stack([dummy_gray, dummy_gray, dummy_gray], axis=-1)

    dummy_color1 = tf.random.uniform([n_test, 224, 224, 3], minval=-1.0, maxval=1.0)
    dummy_color2 = tf.random.uniform([n_test, 224, 224, 3], minval=-1.0, maxval=0.0)
    dummy_color3 = tf.random.uniform([n_test, 224, 224, 3], minval=0.0, maxval=1.0)
    dummy_color = tf.concat([dummy_color1, dummy_color2, dummy_color3], axis=0)

    dummy_img = tf.concat([dummy_gray, dummy_color], axis=0)
    n_test = dummy_img.shape[0]

    close_idx, feat_diff = model.predict(dummy_img, batch_size=64, verbose=1)

    idx_histogram = np.zeros((len(f_names),), dtype=np.int)
    for i in range(n_test):
        idx_histogram[close_idx[i, :n_top]] += 1

    idx_histogram = idx_histogram / n_test

    filter_idx = np.argwhere(idx_histogram > threshold).flatten()
    filter_file_names = f_names[filter_idx]

    with open(filter_file, "w") as f:
        for file_name in filter_file_names:
            f.write(f"{file_name}\n")

