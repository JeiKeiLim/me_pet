from me_pet_python.models import get_base_model
import cv2
import numpy as np
import tensorflow as tf


def webcam_test(img_root, feature_file, filter_file, mp_size=8, base_model=tf.keras.applications.ResNet50V2, gray=True):
    model, f_names = get_base_model(feature_file=feature_file, mp_size=mp_size, base_model=base_model)
    cap = cv2.VideoCapture(0)

    with open(filter_file, 'r') as f:
        filter_names = f.read().split("\n")[:-1]

    if cap.isOpened():
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if gray:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb_frame = cv2.resize(rgb_frame, (224, 224))

            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("window", bgr_frame)

            rgb_frame = (np.array(rgb_frame, dtype=np.float32)/127.5) - 1.0

            close_idx, feat_diff = model.predict(np.expand_dims(rgb_frame, axis=0))
            close_idx, feat_diff = close_idx[0], feat_diff[0]

            i = -1
            plot_idx = 0
            while True:
                i += 1

                if f_names[close_idx[i]] in filter_names:
                    continue

                cv2.imshow(f"pet{plot_idx}", cv2.imread(f"{img_root}/{f_names[close_idx[i]]}"))
                print(f"Dist{i:02d}: {feat_diff[close_idx[i]]:.2f} // {f_names[close_idx[i]]}")

                plot_idx += 1
                if plot_idx > 2:
                    break

            key_in = cv2.waitKey(25) & 0xFF

            if key_in == ord('q'):
                break

        cv2.destroyAllWindows()


