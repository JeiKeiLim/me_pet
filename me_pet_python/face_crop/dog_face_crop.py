import dlib, cv2, os
from tqdm import tqdm
from functools import partial
"""
Code adopted from https://github.com/kairess/dog_face_detector
"""


def save_cropped_face(file_name, face_detector=None, file_root="", save_root="", margin=0):
    file_path = f"{file_root}/{file_name}"
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    det_results = face_detector(image, upsample_num_times=1)

    for i, d in enumerate(det_results):
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()

        x1, y1 = max(x1-margin, 0), max(y1-margin, 0)
        x2, y2 = min(x2+margin, image.shape[1]), min(y2+margin, image.shape[0])

        if d.confidence > 0.8:
            cropped_face = image[y1:y2, x1:x2, :]
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_root}/{file_name}", cropped_face)
            break


def crop_dog_faces(root, target_root, margin=0):
    _, _, file_list = next(os.walk(root))
    file_list = [f for f in file_list if f.endswith(".jpg")]
    detector = dlib.cnn_face_detection_model_v1(f"{os.getcwd()}/face_crop/dogHeadDetector.dat")

    os.makedirs(target_root, exist_ok=True)
    cropper = partial(save_cropped_face, face_detector=detector, file_root=root, save_root=target_root, margin=margin)

    for file in tqdm(file_list, desc="Cropping faces!"):
        cropper(file)
