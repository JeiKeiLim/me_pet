import argparse
from me_pet import *
import tensorflow as tf

if __name__ == "__main__":
    base_model_dict = {
        "resnetv2": tf.keras.applications.ResNet50V2,
        "mobilenetv2": tf.keras.applications.MobileNetV2
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default="test", type=str, help="feature, filter, tfjs, test")
    parser.add_argument("--gray", default=True, dest="gray", action="store_true", help="Convert image to gray for feature extraction")
    parser.add_argument("--no-gray", dest="gray", action="store_false", help="Convert image to gray for feature extraction")
    parser.add_argument("--model", default="resnetv2", type=str, help="resnetv2")
    parser.add_argument("--img-root", default="", type=str, help="Image root directory")
    parser.add_argument("--out-feat", default="./export/feature.csv", type=str, help="Feature result output/input path")
    parser.add_argument("--out-filter", default="./export/filter.txt", type=str, help="Filter name list output/input path")
    parser.add_argument("--out-label", default="./export/labels.txt", type=str, help="Label name list output/input path")
    parser.add_argument("--out-model", default="./export/tf_model", type=str, help="TensorFlow Model output path")
    parser.add_argument("--out-tfjs", default="../tfjs/tfjs_model/", type=str, help="TensorFlow JavaScript Model output path")
    parser.add_argument("--filter-n-top", default=1000, type=int, help="Hyper-parameter for extracting filter labels")
    parser.add_argument("--filter-n-test", default=3000, type=int, help="Hyper-parameter for extracting filter labels")
    parser.add_argument("--filter-threshold", default=0.5, type=float, help="Hyper-parameter for extracting filter labels")
    parser.add_argument("--mp-size", default=8, type=int, help="MaxPooling size")
    args = parser.parse_args()

    base_model = base_model_dict[args.model]

    if args.mode == "feature":
        generate_features(args.img_root, args.out_feat, gray=args.gray, base_model=base_model)
    elif args.mode == "filter":
        generate_filter_list(args.out_feat, args.out_filter, n_top=args.filter_n_top, n_test=args.filter_n_test,
                             threshold=args.filter_threshold, mp_size=args.mp_size, base_model=base_model)
    elif args.mode == "tfjs":
        convert_model_tfjs(args.out_feat, args.out_label, mp_size=args.mp_size, model_root=args.out_model,
                           tfjs_model_root=args.out_tfjs)
    elif args.mode == "test":
        webcam_test(args.img_root, args.out_feat, args.out_filter, mp_size=args.mp_size, base_model=base_model, gray=args.gray)

