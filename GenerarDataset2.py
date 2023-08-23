import cv2
import numpy as np
import pandas as pd
import glob
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

def load_data(label, tipo):
    data = []
    path_pattern = f"microalgasDB/{label}/{tipo}/*"
    
    for file_ in glob.glob(path_pattern):
        img = cv2.imread(file_)
        
        # Convertir a RGB (OpenCV carga las imágenes en BGR por defecto)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aumento de Contraste usando CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        data.append(img)
    
    labels = [label] * len(data)
    return data, labels

def preprocessing(arr):
    preprocessed = []
    for img in arr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = resize(gray, (10, 10), anti_aliasing=True)
        preprocessed.append(resized)
    return preprocessed

def extract_features(arr):
    features = []
    for img in arr:
        hog_features = hog(img, orientations=8, pixels_per_cell=(2, 2),
                           cells_per_block=(1, 1), visualize=False)
        features.append(list(hog_features))
    return features

def generate_dataset():
    data_train_pos, labels_train_pos = load_data('si', 'Train')
    data_train_neg, labels_train_neg = load_data('no', 'Train')
    data_test_pos, labels_test_pos = load_data('si', 'Test')
    data_test_neg, labels_test_neg = load_data('no', 'Test')

    data_train = data_train_pos + data_train_neg
    data_test = data_test_pos + data_test_neg

    labels_train = labels_train_pos + labels_train_neg
    labels_test = labels_test_pos + labels_test_neg

    data_train_processed = preprocessing(data_train)
    data_test_processed = preprocessing(data_test)

    data_train_features = extract_features(data_train_processed)
    data_test_features = extract_features(data_test_processed)

    entire_dataset = data_train_features + data_test_features
    entire_labels = labels_train + labels_test

    df = pd.DataFrame(entire_dataset)
    df['class'] = entire_labels
    df.to_csv("microalgas_dataset.csv", index=False)

if __name__ == '__main__':
    generate_dataset()
