import os
import cv2
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy

def seperate_temp_and_rest_affect(data_path, select):
    temp = []
    rest = []

    list_name_new = []
    list_name = os.listdir(data_path)
    for name in list_name:
        list_name_new.append(os.path.join(data_path, name))

    for im_name in list_name_new:
        if im_name in select.iloc[:, 0].values:
            temp.append(im_name)
        else:
            rest.append(im_name)

    return temp, rest

def ordinary_least_square_affect(img_name, temp):
    img_y = cv2.imread(img_name)
    img_y = cv2.resize(img_y, (100, 100))
    img_y = img_y / 255.
    img_y = img_y.flatten()
    img_mat = []
    for i in range(len(temp)):
        img_temp = cv2.imread(temp[i])
        img_temp = cv2.resize(img_temp, (100, 100))
        img_temp = img_temp / 255.
        img_temp = img_temp.flatten().tolist()
        img_mat.append(img_temp)
    img_mat = DataFrame(img_mat)
    img_x = img_mat.T

    model = LinearRegression()
    model.fit(img_x, img_y)
    a = model.coef_
    b = model.intercept_

    img_predict = np.dot(img_x, a) + b
    img_predict = (img_predict - np.min(img_predict)) / (np.max(img_predict) - np.min(img_predict))
    img_predict = (img_predict * 255).astype("uint8")

    return img_predict.reshape(100, 100, 3)

if __name__ == '__main__':
    category = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    path = '/data1/conceptual_affect'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    for cat in category:
        os.makedirs(os.path.join(path, cat), exist_ok=True)
        data_path = f'/data1/Affect-net/train/{cat}'
        select = pd.read_csv(f'representative_affect/{cat}_select.csv')
        temp, rest = seperate_temp_and_rest_affect(data_path, select)

        for i, img_name in enumerate(rest):
            index = img_name.index(cat)
            img_predict = ordinary_least_square_affect(img_name, temp)
            cv2.imwrite(os.path.join(path, cat, f'{img_name[index+len(cat)+1:-4]}_con.jpg'), img_predict)
            print(cat, "rest:", len(rest), "process:", f'{i+1}/{len(rest)}')

        for i, img_name in enumerate(temp):
            index = img_name.index(cat)
            to_path = os.path.join(path, cat, f'{img_name[index+len(cat)+1:-4]}_con.jpg')
            copy(img_name, to_path)
            print(cat, "temp:", len(temp), "process:", f'{i+1}/{len(temp)}')

# cat = 'sad'
# data_path = f'/data1/Affect-net/train/{cat}'
# select = pd.read_csv(f'representative_affect/{cat}_select.csv')
# temp, rest = seperate_temp_and_rest_affect(data_path, select)
#
# for i, img_name in enumerate(rest):
#     print(img_name)
#     index = img_name.index(cat)
#     print(index)
#     print(img_name[index+len(cat)+1:-4])
#     break