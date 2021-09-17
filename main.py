'''
Leon's main.py   JLL, 2021.8.14 - 9.5, 9.17

1. Download modeld and main.py (see https://github.com/JinnAIGroup/OPNet)
2. mv /dataA/.../fcamera.hevc to /Leon/fcamera.hevc
3. (YPN) jinn@Liu:~/YPN/Leon$ python3 main.py ./fcamera.hevc
4. Read output.txt
5. Your Project: Build your own Net to replace supercombo.h5.
'''
import sys
import cv2   # https://cs.gmu.edu/~kosecka/cs482/code-examples/opencv-python/OpenCV_Python.pdf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser

camerafile = sys.argv[1]
supercombo = load_model('models/supercombo.keras')
# OK in loading and running
#supercombo = load_model('models/opEffPNb2.h5')
# OK in loading, NG in running
# ValueError: Unknown loss function: custom_loss
'''
supercombo = load_model('models/opUNet1.h5')
# OK in loading
supercombo = load_model('models/yolov3.h5')
OK in loading
supercombo = load_model('models/mobile.h5')
OK in loading
supercombo = load_model('models/opEffA2.h5')
NG in loading
supercombo = load_model('models/mnist.h5')
NG in loading
ValueError: No model found in config file.
supercombo = load_model('models/open.h5')
NG in loading
ValueError: No model found in config file.
supercombo = load_model('models/opEffA.h5')
NG in loading
ValueError: Unknown layer: MBConvBlock
supercombo = load_model('models/masksup3.h5')
NG in loading
ValueError: bad marshal data (unknown type code)
supercombo = load_model('models/weights-opeffA.best.hdf5')
NG in loading
ValueError: Unknown layer: MBConvBlock
  File "/home/jinn/.pyenv/versions/YPN/lib/python3.8/site-packages/tensorflow/python/keras/saving/save.py", line 182, in load_model
    return hdf5_format.load_model_from_hdf5(filepath, custom_objects, compile)
'''

cap = cv2.VideoCapture(camerafile)

imgs = []

#for i in tqdm(range(1000)):
for i in tqdm(range(1000//50)):
  ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  if i==0:
    print("\n---JLL   ret = ", ret)
    print("---JLL   frame.shape = ", frame.shape) # 874*1164*3
    print("---JLL   img_yuv.shape = ", img_yuv.shape) # 1311×1164 = 1526004
    x = img_yuv.reshape((874*3//2, 1164))
    print("---JLL   img_yuv.reshape = ", x.shape)
  imgs.append(img_yuv.reshape((874*3//2, 1164))) # 874*3//2 = 1311
  # http://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf
  # https://drive.google.com/file/d/1tWSU4Y-xUSI-sy6ht2wfeF-w687k_Y9D/view
  # https://en.wikipedia.org/wiki/Camera_resectioning

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8) # np.uint8 = 0~255
print("---JLL   imgs.shape = ", np.shape(imgs))
print("---JLL   imgs_med_model.shape = ", imgs_med_model.shape)

for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                                yuv=True, output_size=(512,256))
  # transform_img.cv2.COLOR_YUV2RGB_I420 => YUV img (input) to RGB imgs_med_model (output)

def frames_to_tensor(frames):
  H = (frames.shape[1]*2)//3  # 384x2//3 = 768//3 = 256
  W = frames.shape[2]         # 512
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

  in_img1[:, 0] = frames[:, 0:H:2, 0::2]
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0   # /128.0 - 1.0?
print("---JLL   np.array(imgs_med_model).shape = ", np.shape(np.array(imgs_med_model)))
print("---JLL   frame_tensors.shape = ", np.shape(frame_tensors))

state = np.zeros((1,512))
desire = np.zeros((1,8))

cap = cv2.VideoCapture(camerafile)

for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
  outs = supercombo.predict(inputs)
  parsed = parser(outs)
  # Important to refeed the state
  state = outs[-1]
  pose = outs[-2]
  ret, frame = cap.read()
  frame = cv2.resize(frame, (640, 420))
  # Show raw camera image
  if i==0:
    #print("---JLL   inputs = ", inputs)
    print("\n")
    print("---JLL   frame_tensors[i:i+2].shape = ", np.shape(frame_tensors[i:i+2]))
    print("---JLL   np.vstack(frame_tensors[i:i+2]).shape = ", np.shape(np.vstack(frame_tensors[i:i+2])))
    print("---JLL   np.vstack(frame_tensors[i:i+2])[None].shape = ", np.shape(np.vstack(frame_tensors[i:i+2])[None]))
    [print("---JLL   inputs[", i, "].shape = ", np.shape(inputs[i])) for i in range(len(inputs))]
    #print("---JLL     outs = ", outs)
    [print("---JLL   outs[", i, "].shape = ", np.shape(outs[i])) for i in range(len(outs))]
    #print("---JLL   parsed = ", parsed)
    [print("---JLL   parsed[", x, "].shape = ", parsed[x].shape) for x in parsed]
    print("---JLL    state.shape = ", np.shape(state))
    print("---JLL     pose.shape = ", np.shape(pose))
    print("---JLL   frame.cv2.resize.shape = ", frame.shape)
  cv2.imshow("modeld", frame)
  # Clean plot for next frame
  plt.clf()
  plt.title("lanes and path")
  # lll = left lane line
  plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
  # rll = right lane line
  plt.plot(parsed["rll"][0], range(0,192), "r-", linewidth=1)
  # path = path cool isn't it ?
  plt.plot(parsed["path"][0], range(0,192), "g-", linewidth=1)
  #print(np.array(pose[0,:3]).shape)
  #plt.scatter(pose[0,:3], range(3), c="y")

  # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  plt.gca().invert_xaxis()
  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

input("Press ENTER twice to close all windows ...")
input("Press ENTER to exit ...")
# pauses for 1 second
if cv2.waitKey(1000) == 27: #if ENTER is pressed
  cap.release()
  cv2.destroyAllWindows()
plt.pause(0.5)
plt.close()
#plt.show()
