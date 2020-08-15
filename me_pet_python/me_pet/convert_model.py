from models import get_base_model
import tensorflowjs as tfjs
import tensorflow as tf
import re


def convert_model_tfjs(feature_file, label_name, mp_size=8, base_model=tf.keras.applications.ResNet50V2,
                       model_root="./export/tf_model", tfjs_model_root="../docs/tfjs_model/"):
    model, f_names = get_base_model(feature_file=feature_file, mp_size=mp_size, base_model=base_model)

    model.save(model_root, save_format="tf")
    tfjs.converters.convert_tf_saved_model(model_root, tfjs_model_root)

    with open(label_name, "w") as f:
        for f_name in f_names:
            f.write(f"{f_name}\n")

    unique_names = list(set([re.sub("_[0-9]+\..*", "", f_names[i]) for i in range(len(f_names))]))
    with open(f"{label_name[:-4]}_unique.txt", 'w') as f:
        for f_name in unique_names:
            f.write(f"{f_name}\n")

