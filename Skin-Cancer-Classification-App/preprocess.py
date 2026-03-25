import cv2
import os
import numpy as np

IMG_SIZE = 100

def load_images(data_dir):
    data = []
    labels = []

    for label in os.listdir(data_dir):   # ✅ correct
        folder_path = os.path.join(data_dir, label)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                data.append(img)
                labels.append(label)

            except:
                continue

    return np.array(data), np.array(labels)


