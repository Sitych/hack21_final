import numpy as np
import cv2
from PIL import Image, ImageDraw
from multiprocessing import Pool
import multiprocessing
import time
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


def img_show(img, text, imshow):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 10
    fontColor              = (0,0,255)
    lineType               = 2

    cv2.putText(img, text,
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow(imshow,img)


def cool_func(frame):
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    x,y,w,h = faces[0] 
    img = frame[y:y+np.int(h/3), x:x+w]
    img_show(img, '', 'lob')
    arrayMeanForehead = np.average(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return str(int(np.mean(arrayMeanForehead)))
  
pathMove = '/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/Для теста/Видео для теста.MOV'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

vidcap = cv2.VideoCapture(pathMove)
success,frame = vidcap.read()

count = 0
text = ''
def frame2x(frame):
    width = frame.shape[0]
    height = frame.shape[1]
    for i in range(width):
        for j in range(height):
            frame[i,j] = np.mean(frame[i,j])
    return frame.reshape(1, frame.size)[:1822]

def load_model(x):
    import random
    return random.randint(1, 3)
text_plus = ''
while (vidcap.isOpened()):
    if count % 30 == 0:
        class_from_model = load_model(1)
        if class_from_model == 1:
            text_plus = 'normal pulse'
        elif class_from_model == 2:
            text_plus = 'rapid pulse'
        elif class_from_model == 3:
            text_plus = 'very rapid pulse'
        
        text = cool_func(frame)
    count += 1
    img_show(frame, text + '_' + str(text_plus), 'frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success,frame = vidcap.read()
cap.release()
cv2.destroyAllWindows()

# while success:
#     faces = face_cascade.detectMultiScale(frame, 1.3, 5)
#     for faces in results:
#         x,y,w,h = faces[0] 
#         img = frame[y:y+np.int(h/3), x:x+w]
#         arrayMeanForehead[i] = np.average(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#         i = i +1
#     success,frame = vidcap.read()

import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/video_cat/видео по 1 шаблону пульс 90-80.mov')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()