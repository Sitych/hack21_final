import json
import base64
import io
from PIL import Image, ImageDraw
import yaml
from model_loader import ModelLoader
import logging as logg
import numpy as np
import cv2
import h5py
import os

COL_IN_SEC = 5

def from_rgb_to_gray(image):
    # draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования
    # width = image.size[0]  # Определяем ширину
    # height = image.size[1]  # Определяем высоту
    # pix = image.load()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            r = image[x, y][0] #узнаём значение красного цвета пикселя
            g = image[x, y][1] #зелёного
            b = image[x, y][2] #синего
            sr = (r + g + b) // 3 #среднее значение
            draw.point((x, y), (sr, sr, sr)) #рисуем пиксель

def jpg_image_to_array(image):
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr

def crop_man(im2arr):
    im = Image.fromarray(im2arr)
    (boxes, scores, classes, num_detections) = context.infer(im)
    xtl = boxes[0][0][1] * im.width
    ytl = boxes[0][0][0] * im.height
    xbr = boxes[0][0][3] * im.width
    ybr = boxes[0][0][2] * im.height / (3 / 2)
    im2arr = im2arr[int(ytl):int(ybr)]
    im2arr = im2arr[:,int(xtl):int(xbr)]
    # from_rgb_to_gray(im2arr)
    # im2arr = jpg_image_to_array(im2arr)
    return im2arr

def from_video_to_h5py(video_path, h5py_name):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()

    count = 0
    i = 0
    dst_to_target = 'target_1'
    os.makedirs(dst_to_target, exist_ok=True)
    with h5py.File(h5py_name + '-1' + '.h5py', 'w') as f:
        while success:
            if count % COL_IN_SEC == 0:
                image_name = f"frame_{i}"
                image_arr = crop_man(image)
                files_path = os.path.join(dst_to_target, os.path.basename(video_path).split('.')[0])
                os.makedirs(files_path, exist_ok=True)
                image_path = os.path.join(files_path, image_name + '.png')
                cv2.imwrite(image_path, image_arr)
                i += 1
            count += 1
            success,image = vidcap.read()
            if not success:
                break
    print(f"SUCCES LOAD: {video_path}")

def from_video_to_pics(video_path, h5py_name):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()

    count = 0
    i = 0
    dst_to_target = 'target_1'
    os.makedirs(dst_to_target, exist_ok=True)
    while success:
        if count % COL_IN_SEC == 0:
            image_name = f"frame_{i}"
            files_path = os.path.join(dst_to_target, os.path.basename(video_path).split('.')[0])
            os.makedirs(files_path, exist_ok=True)
            image_path = os.path.join(files_path, image_name + '.png')
            cv2.imwrite(image_path, image)
            i += 1
            exit()
        count += 1
        success,image = vidcap.read()
        if not success:
            break
    print(f"SUCCES LOAD: {video_path}")

model_path='/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
videos_path = 'Для теста'
abs_video_path = os.path.abspath(videos_path)

context = ModelLoader(model_path=model_path)

mov_file = '/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/Для теста/Видео для теста.MOV'

mov_path = os.path.join(abs_video_path, os.path.basename(mov_file))
h5py_name = mov_file.split('.')[0]
from_video_to_h5py(mov_path, h5py_name)

