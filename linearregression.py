import numpy as np
import os
from sklearn.linear_model import LinearRegression
from skimage import io
import pandas as pd
from pandas.core.frame import DataFrame
import math
import cv2
from shutil import copy


def seperate_temp_and_rest(landmarks, select):
    temp = []
    rest = []
    for i in range(len(landmarks)):
        if landmarks.iloc[i, 0] in select['representative images'].values:
            sample = np.array(landmarks.iloc[i, 1:]).flatten().tolist()
            sample.insert(0, landmarks.iloc[i, 0])
            temp.append(sample)
        else:
            sample = np.array(landmarks.iloc[i, 1:]).flatten().tolist()
            sample.insert(0, landmarks.iloc[i, 0])
            rest.append(sample)

    return DataFrame(temp), DataFrame(rest)


def dx_dy(ld):
    ld = np.array(ld).reshape(-1, 2)
    lefteye = ld[36:42]
    righteye = ld[42:48]
    lefteyecenter = lefteye.mean(axis=0).astype("int")
    righteyecenter = righteye.mean(axis=0).astype("int")
    dY = righteyecenter[1] - lefteyecenter[1]
    dX = righteyecenter[0] - lefteyecenter[0]
    return dY, dX

def scale_angle(dY_target, dX_target, dY_temp, dX_temp):
    angle_target = math.atan2(dY_target, dX_target)
    angle_target = int(angle_target * 180 / math.pi)
    dist_target = np.sqrt((dX_target ** 2) + (dY_target ** 2))

    angle_temp = math.atan2(dY_temp, dX_temp)
    angle_temp = int(angle_temp * 180 / math.pi)
    dist_temp = np.sqrt((dX_temp ** 2) + (dY_temp ** 2))

    if angle_target * angle_temp >= 0:
        insideAngle = abs(angle_target - angle_temp)
    else:
        insideAngle = abs(angle_target) + abs(angle_temp)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    scale = dist_target / dist_temp
    return scale, insideAngle

# temp
def temp_mat(dir_temp, temp, affine=True):
    img_mat = []
    for i in range(len(temp)):
        img_temp = io.imread(os.path.join(dir_temp, temp.iloc[i, 0]))
        if affine:
            ld_temp = temp.iloc[i, 1:]
            dY_temp, dX_temp = dx_dy(ld_temp)
            scale, insideAngle = scale_angle(dY_target, dX_target, dY_temp, dX_temp)
            noseCenter = np.array(ld_temp).reshape(-1, 2)[30]
            noseCenter = (noseCenter[0], noseCenter[1])

            M = cv2.getRotationMatrix2D(noseCenter, insideAngle, scale)

            (w, h) = (100, 100)
            img_x = cv2.warpAffine(img_temp, M, (w, h), flags=cv2.INTER_CUBIC)

        else:
            img_x = img_temp

        img_x = img_x / 255.
        img_x = img_x.flatten().tolist()
        img_mat.append(img_x)

    return DataFrame(img_mat)

def ordinary_least_square(dir, img_name, dir_temp, temp):
    img_y = io.imread(os.path.join(dir, img_name))
    img_y = img_y / 255.
    img_y = img_y.flatten()
    img_mat = temp_mat(dir_temp, temp, affine=False)
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
    path = '/data1/conceptual_happy_destroy'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    # facial landmarks
    # for cat in category:
    #     os.makedirs(os.path.join(path, cat), exist_ok=True)
    #     landmarks = pd.read_csv(f'landmarks/{cat}_68.csv')
    #     # representative faces
    #     select = pd.read_csv(f'representative_faces/{cat}_select.csv')
    #     dir = f'../RAF-DB/train/{cat}'
    #     temp, rest = seperate_temp_and_rest(landmarks, select)
    #
    #     for i in range(len(temp)):
    #         img_name = temp.iloc[i, 0]
    #         from_path = os.path.join(dir, img_name)
    #         to_path = os.path.join(path, cat, f'{img_name[:-4]}_con.jpg')
    #         copy(from_path, to_path)
    #
    #     for i in range(len(rest)):
    #         ld_target = rest.iloc[i, 1:]
    #         dY_target, dX_target = dx_dy(ld_target)
    #         img_name = rest.iloc[i, 0]
    #         img_predict = ordinary_least_square(dir, img_name, temp)
    #
    #         io.imsave(f'{path}/{cat}/{img_name[:-4]}_con.jpg', img_predict)
    #         print(f"{img_name[:-4]}_con.jpg is completed")

    # create sad temp for happy images
    cat_h = 'happy'
    cat_s = 'sad'
    os.makedirs(os.path.join(path, cat_h), exist_ok=True)
    landmarks_s = pd.read_csv(f'landmarks/{cat_s}_68.csv')
    select_s = pd.read_csv(f'representative_faces/{cat_s}_select.csv')
    dir = f'../RAF-DB/train/{cat_h}'
    dir_temp = f'../RAF-DB/train/{cat_s}'
    temp, _ = seperate_temp_and_rest(landmarks_s, select_s)

    landmarks_h = pd.read_csv(f'landmarks/{cat_h}_68.csv')
    select_h = pd.read_csv(f'representative_faces/{cat_h}_select.csv')
    temp_h, rest_h = seperate_temp_and_rest(landmarks_h, select_h)
    happy = pd.concat([temp_h, rest_h], axis=0)

    exist_im = os.listdir(os.path.join(path, cat_h))
    sum = 0
    for i in range(len(happy)):
        if f'{happy.iloc[i, 0][:-4]}_con.jpg' in exist_im:
            continue
        else:
            ld_target = happy.iloc[i, 1:]
            dY_target, dX_target = dx_dy(ld_target)
            img_name = happy.iloc[i, 0]
            # img_truth = io.imread(os.path.join(dir, img_name))
            img_predict = ordinary_least_square(dir, img_name, dir_temp, temp)
            io.imsave(f'{path}/{cat_h}/{img_name[:-4]}_con.jpg', img_predict)
            print(f'{i+1}/4691')



















