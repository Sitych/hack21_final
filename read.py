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

with h5py.File('first_video.h5py', 'r') as f:
   data_set = f['photos/frame0']
   Image.fromarray(data_set).show()

