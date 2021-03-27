from PIL import Image, ImageDraw
from model_loader import ModelLoader
import numpy as np
import cv2

vidcap = cv2.VideoCapture('/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/video_cat/видео по 1 шаблону пульс 90-80.mov')
context = ModelLoader(model_path='/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
success,image = vidcap.read()
im = Image.fromarray(image)
(boxes, scores, classes, num_detections) = context.infer(im)

for i in range(int(num_detections[0])):
    xtl = boxes[0][i][1] * im.width
    ytl = boxes[0][i][0] * im.height
    xbr = boxes[0][i][3] * im.width
    ybr = boxes[0][i][2] * im.height

    shape = [xtl, ytl, xbr, ybr]
    img1 = ImageDraw.Draw(im)
    img1.rectangle(shape, outline ="red", width=5)
    im.save("lol.jpg", 'JPEG')
    # del it if want save more than 1 photo