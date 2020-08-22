import argparse
from me_pet import *
from face_crop import crop_dog_faces
import tensorflow as tf
import json

if __name__ == "__main__":
    base_model_dict = {
        "resnetv2": tf.keras.applications.ResNet50V2,
        "mobilenetv2": tf.keras.applications.MobileNetV2
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default="test", type=str, help="feature, filter, tfjs, test")
    parser.add_argument("--conf", default="./config.json", type=str, help="Configuration file path")
    parser.add_argument("-co", "--crop-out", default="./export/cropped_img", type=str, help="Output root of crop faces")
    parser.add_argument("-cm", "--crop-margin", default=0, type=int, help="Crop margin size")
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    base_model = base_model_dict[conf['model']]

    if args.mode == "feature":
        generate_features(conf['img_root'], conf['out_feat'], gray=conf['gray'], base_model=base_model)
    elif args.mode == "filter":
        generate_filter_list(conf['out_feat'], conf['out_filter'], n_top=conf['filter_n_top'], n_test=conf['filter_n_test'],
                             threshold=conf['filter_threshold'], mp_size=conf['mp_size'], base_model=base_model)
    elif args.mode == "tfjs":
        convert_model_tfjs(conf['out_feat'], conf['out_label'], mp_size=conf['mp_size'], model_root=conf['out_model'],
                           tfjs_model_root=conf['out_tfjs'], base_model=base_model)
    elif args.mode == "crop":
        crop_dog_faces(conf['img_root'], args.crop_out, margin=args.crop_margin)
    elif args.mode == "test":
        webcam_test(conf['img_root'], conf['out_feat'], conf['out_filter'], mp_size=conf['mp_size'],
                    base_model=base_model, gray=conf['gray'])
