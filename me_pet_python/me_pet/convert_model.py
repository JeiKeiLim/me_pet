from models import get_base_model
import tensorflowjs as tfjs


def convert_model_tfjs(feature_file, label_name, mp_size=8, model_root="./export/tf_model", tfjs_model_root="../tfjs/tfjs_model/"):
    model, f_names = get_base_model(feature_file=feature_file, mp_size=mp_size)

    model.save(model_root, save_format="tf")
    tfjs.converters.convert_tf_saved_model(model_root, tfjs_model_root)

    with open(label_name, "w") as f:
        for f_name in f_names:
            f.write(f"{f_name}\n")

