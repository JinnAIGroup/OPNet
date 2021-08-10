#!/usr/bin/env python3
# For Colab by JLL 2020.6.12-19

# 'import module' vs. 'from module import xyz'
# xyz can be a module, subpackage, or object (class or function)
# 'import sys' binds the name sys to the module (so sys -> sys.modules['sys']), 
# 'from sys import argv' binds a different name, argv, pointing at the attribute argv in sys 
# (so argv -> sys.modules['sys'].argv).
# foo.py
# mylib\
#    a.py
#    b.py
# import b.py into a.py then import a.py to foo
# In a.py, write 'import b'. In foo.py, write 'from mylib import b'.

import cv2 
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

from tools.lib.parser import parser
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics

#####Colab
import os
import plotly.graph_objects as pltCo
from tensorflow import keras
from google.colab.patches import cv2_imshow
#####Colab

camerafile = sys.argv[1]
print("OK. This is the name of the script: ", sys.argv[1])  #Colab
supercombo = load_model('supercombo.keras', compile=False)

# Get architecture of supercombo
from contextlib import redirect_stdout
with open('supercomboArc.txt', 'w') as f:
    with redirect_stdout(f):
        supercombo.summary()
        
MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

cap = cv2.VideoCapture(camerafile)

imgs = []

for i in tqdm(range(1000//100)):
#for i in tqdm(range(1000)):
  ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  imgs.append(img_yuv.reshape((874*3//2, 1164)))
 
print("OK. for i in tqdm(range(1000)): ") #Colab

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0

state = np.zeros((1,512))
desire = np.zeros((1,8))

cap = cv2.VideoCapture(camerafile)
print("cap = cv2.VideoCapture(camerafile) OK ")

###Colab
res=(640,420) #resulotion
fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
out = cv2.VideoWriter('videoOPNet.mp4', fourcc, 20.0, res)
###Colab

for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
# Train model with model.fit(). Use it to predict with model.predict()
  outs = supercombo.predict(inputs)

# Get outputs from a layer in supercombo, i.e., 
# extract features into a dataset from keras model
  desire_input = supercombo.input
  #desire_output = supercombo.get_layer('snpe_desire_pleaser').output
  desire_output = supercombo.get_layer('vision_vision_features').output
  #print("OK. get_layer('snpe_desire_pleaser').output: ", desire_output)  #Colab
  extract = keras.Model(desire_input, desire_output)
  features = extract.predict(inputs)
  print("OK. features = ", features)  #Colab

  parsed = parser(outs)
  # Important to refeed the state
  state = outs[-1]
  pose = outs[-2]
  ret, frame = cap.read()
  frame = cv2.resize(frame, (640, 420))
  # Show raw camera image
#  cv2.imshow("modeld", frame)
#  cv2_imshow(frame) #NG, Colab
  out.write(frame)   #Colab
  # Clean plot for next frame
  plt.clf()
  plt.title("lanes and path")
  # lll = left lane line
  plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
  # rll = right lane line
  plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=1)
  # path = path cool isn't it ?
  plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)
  #print(np.array(pose[0,:3]).shape)
  #plt.scatter(pose[0,:3], range(3), c="y")
  
  # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  plt.gca().invert_xaxis()
  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

plt.show()

#####Colab
#path, ll, rl, lead, long_x, long_v, long_a, desire_state, meta, desire_pred, pose=outs
#print(path)
cur_dir = os.getcwd()
print("videoOPNet.mp4 in: ", cur_dir)
out.release()
print("End of main.py: OK ")
#####Colab
