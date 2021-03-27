from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

COL_IN_SEC = 15

def jpg_image_to_array(image):
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr

def from_video_to_pics(video_path, h5py_name):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()

    count = 0
    i = 0
    dst_to_target = 'target'
    os.makedirs(dst_to_target, exist_ok=True)
    while success:
        if count % COL_IN_SEC == 0:
            image_name = f"frame_{i}"
            files_path = os.path.join(dst_to_target, os.path.basename(video_path).split('.')[0])
            os.makedirs(files_path, exist_ok=True)
            image_path = os.path.join(files_path, image_name + '.jpeg')
            cv2.imwrite(image_path, image)
            i += 1
        count += 1
        success,image = vidcap.read()
        if not success:
            break
    print(f"SUCCES LOAD: {video_path}")

model_path='/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
videos_path = 'video_cat'
abs_video_path = os.path.abspath(videos_path)

for level in os.walk(abs_video_path):
    for mov_file in level[2]:
        mov_path = os.path.join(abs_video_path, os.path.basename(mov_file))
        h5py_name = mov_file.split('.')[0]
        from_video_to_pics(mov_path, h5py_name)
