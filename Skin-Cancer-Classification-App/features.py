import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(images):
    feature_list = []

    for img in images:
        # Basic features
        mean = np.mean(img)
        std = np.std(img)
        max_val = np.max(img)
        min_val = np.min(img)

        # GLCM (texture features 🔥)
        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        feature_list.append([
            mean, std, max_val, min_val,
            contrast, energy, homogeneity
        ])

    return np.array(feature_list)