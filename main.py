'''  JLL, 2021.8.14, 9.5, 9.21
Leon's main.py

1. Download modeld and main.py (see https://github.com/JinnAIGroup/OPNet)
2. mv /dataA/.../fcamera.hevc to /Leon/fcamera.hevc
3. (YPN) jinn@Liu:~/YPN/Leon$ python main.py ./fcamera.hevc
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
'''
supercombo = load_model('models/opEffPNb2.h5')
OK in loading, NG in running
ValueError: Unknown loss function: custom_loss
'''

cap = cv2.VideoCapture(camerafile)
imgs = []

#for i in tqdm(range(1000)):
for i in tqdm(range(10)):
  ret, frame = cap.read()
  #---  ret =  True
  #---  frame.shape = (874, 1164, 3) = (H, W, C) = (row, col, dep) = (axis0, axis1, axis2)
  # Total float numbers (TFNs) = 874*1164*3
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  imgs.append(img_yuv.reshape((874*3//2, 1164))) # 874*3//2 = 1311
  #if i==0:
    #x = img_yuv.reshape((874*3//2, 1164))
    #---  img_yuv.shape = (1311, 1164) # TFNs = 874*1164*3/2
    #---  x.shape = (1311, 1164)

#---  np.shape(imgs) = (20, 1311, 1164)
imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8) # np.uint8 = 0~255
#---  imgs_med_model.shape = (20, 384, 512)

for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                                    yuv=True, output_size=(512, 256))  # (W, H)
'''
input: YUV img (1311, 1164) => transform_img() => cv2.COLOR_YUV2RGB_I420
  => RGB (874, 1164, 3) => cv2.warpPerspective) => np.clip
  => RGB (256, 512, 3) => cv2.COLOR_RGB2YUV_I420 =>
output: YUV imgs_med_model[0].shape = (384, 512)  # 256*3//2 = 384
RGB, YUV444: 3 bytes per pixel; YUV420: 6 bytes per 4 pixels [wiki yuv]
RGB (874, 1164, 3) = 874*1164*3 bytes => YUV (1311, 1164) = 1311*1164 = 874*3//2*1164 bytes
'''

def frames_to_tensor(frames):
  #---  np.shape(frames) = (20, 384, 512) = (B, H, W)  YUV
  H = (frames.shape[1]*2)//3  # 384x2//3 = 256
  W = frames.shape[2]         # 512
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

  in_img1[:, 0] = frames[:, 0:H:2, 0::2]
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))

  return in_img1
'''
np.shape(in_img1) = (20, 6, 128, 256) = (B, C, H, W) YUV420 => C = 6 ???
RGB (256, 512, 3) = 256*512*3 bytes => YUV (384, 512) = 256*3//2*512
= 128*512*3 = 128*256*6 bytes => C = 6 QED
'''

frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0   # /128.0 - 1.0?
#---  np.shape(np.array(imgs_med_model)) =  (20, 384, 512)
#---  np.shape(frame_tensors) = (20, 6, 128, 256)

state = np.zeros((1,512))
desire = np.zeros((1,8))

print("#---  Input: camerafile = ", camerafile)
print("#---  Input: np.shape(frame_tensors) = ", np.shape(frame_tensors))

for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
  #if i==0:
    #---  inputs[ 0 ].shape = (1, 12, 128, 256)
    #---  inputs[ 1 ].shape = (1, 8)
    #---  inputs[ 2 ].shape = (1, 512)
  outs = supercombo.predict(inputs)
    #---  outs[ 0 ].shape = (1, 385)
    #---  outs[ 1 ].shape = (1, 386)
    #---  outs[ 2 ].shape = (1, 386)
    #---  outs[ 3 ].shape = (1, 58)
    #---  outs[ 4 ].shape = (1, 200)
    #---  outs[ 5 ].shape = (1, 200)
    #---  outs[ 6 ].shape = (1, 200)
    #---  outs[ 7 ].shape = (1, 4)
    #---  outs[ 8 ].shape = (1, 32)
    #---  outs[ 9 ].shape = (1, 12)
    #---  outs[ 10 ].shape = (1, 512)
  parsed = parser(outs)
    #---  parsed[ path ].shape = (1, 192)
    #---  parsed[ path_stds ].shape = (1, 192)
    #---  parsed[ lll ].shape = (1, 192)
    #---  parsed[ lll_prob ].shape = (1,)
    #---  parsed[ lll_stds ].shape = (1, 192)
    #---  parsed[ rll ].shape = (1, 192)
    #---  parsed[ rll_prob ].shape = (1,)
    #---  parsed[ rll_stds ].shape = (1, 192)
    #---  parsed[ lead_xyva ].shape = (1, 4)
    #---  parsed[ lead_xyva_std ].shape = (1, 4)
    #---  parsed[ lead_prob ].shape = (1,)
    #---  parsed[ lead_xyva_2s ].shape = (1, 4)
    #---  parsed[ lead_xyva_std_2s ].shape = (1, 4)
    #---  parsed[ lead_prob_2s ].shape = (1,)
    #---  parsed[ lead_all ].shape = (1, 58)
    #---  parsed[ meta ].shape = (1, 32)
    #---  parsed[ desire ].shape = (1, 12)
    #---  parsed[ desire_state ].shape = (1, 4)
    #---  parsed[ long_x ].shape = (1, 200)
    #---  parsed[ long_v ].shape = (1, 200)
    #---  parsed[ long_a ].shape = (1, 200)
    #---  parsed[ trans ].shape = (1, 3)
    #---  parsed[ trans_std ].shape = (1, 3)
    #---  parsed[ rot ].shape = (1, 3)
    #---  parsed[ rot_std ].shape = (1, 3)
  # Important to refeed the state
  state = outs[-1]
  #---  np.shape(state) = (1, 512)
  pose = outs[-2]
  #print(np.array(pose[0,:3]).shape)
  #---  np.shape(pose) = (1, 12)
  ret, frame = cap.read()
  #---  frame.shape = (874, 1164, 3) = (H, W, C)
  frame = cv2.resize(frame, (640, 420))
  #---  frame.shape = (420, 640, 3) = (H, W, C)

  # Show raw camera image
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

'''
OPNet/camera.py:
FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

  if i==0:
    #print("#---  inputs = ", inputs)
    print("#---  frame_tensors[i:i+2].shape =", np.shape(frame_tensors[i:i+2]))
    print("#---  np.vstack(frame_tensors[i:i+2]).shape =", np.shape(np.vstack(frame_tensors[i:i+2])))
    print("#---  np.vstack(frame_tensors[i:i+2])[None].shape =", np.shape(np.vstack(frame_tensors[i:i+2])[None]))
    [print("#---  inputs[", i, "].shape =", np.shape(inputs[i])) for i in range(len(inputs))]
    #print("#---  outs =", outs)
    [print("#---  outs[", i, "].shape =", np.shape(outs[i])) for i in range(len(outs))]
    #print("#---  parsed =", parsed)
    [print("#---  parsed[", x, "].shape =", parsed[x].shape) for x in parsed]
    print("#---  np.shape(state) =", np.shape(state))
    print("#---  np.shape(pose) =", np.shape(pose))
    print("#---  frame.shape =", frame.shape)

supercombo = load_model('models/opEffPNb2.h5')
OK in loading, NG in running
ValueError: Unknown loss function: custom_loss
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
