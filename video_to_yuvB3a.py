'''  JLL, 2021.9.24, 10.2, 10.6
From /home/jinn/YPN/Leon/video_to_yuv.py

(OP082) jinn@Liu:~/openpilot/tools/lib$ python video_to_yuvB3a.py
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.lib.framereader import FrameReader
from cameraB3 import transform_img, eon_intrinsics
# cameraB3 = /home/jinn/YPN/Leon/common/transformations/camera.py
from common.transformations.model import medmodel_intrinsics
'''
big   RGB (874, 1164, 3) = (H, W, C) => big YUV (1311, 1164) = (X, Y) =>
small YUV (384, 512) => CsYUV (6, 128, 256) = (C, x, y) =>
small RGB (256, 512, 3)
'''
def RGB_to_sYUVs(video, frame_count):
  bYUVs = []

  #for i in tqdm(range(10)):
  for i in range(frame_count):
    ret, frame = video.read()  #---  ret =  True
    #---  frame.shape = (874, 1164, 3) = (H, W, C) = (row, col, dep) = (axis0, axis1, axis2)
    bYUV = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
    #---  np.shape(bYUV) = (1311, 1164)
    bYUVs.append(bYUV.reshape((874*3//2, 1164))) # 874*3//2 = 1311

  #---  bYUV.shape = (1311, 1164) # TFNs = 874*1164*3/2 bytes
  #---  np.shape(bYUVs) = (10, 1311, 1164)
  sYUVs = np.zeros((len(bYUVs), 384, 512), dtype=np.uint8) # np.uint8 = 0~255

  for i, img in tqdm(enumerate(bYUVs)):
    sYUVs[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                             yuv=True, output_size=(512, 256))  # (W, H)

  print("#---  np.shape(sYUVs) =", np.shape(sYUVs))
  return sYUVs

def sYUVs_to_CsYUVs(sYUVs):
  H = (sYUVs.shape[1]*2)//3  # 384x2//3 = 256
  W = sYUVs.shape[2]         # 512
  CsYUVs = np.zeros((sYUVs.shape[0], 6, H//2, W//2), dtype=np.uint8)

  CsYUVs[:, 0] = sYUVs[:, 0:H:2, 0::2]  # [2::2] get every even starting at 2
  CsYUVs[:, 1] = sYUVs[:, 1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
  CsYUVs[:, 2] = sYUVs[:, 0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
  CsYUVs[:, 3] = sYUVs[:, 1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
  CsYUVs[:, 4] = sYUVs[:, H:H+H//4].reshape((-1, H//2, W//2))
  CsYUVs[:, 5] = sYUVs[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))

  CsYUVs = np.array(CsYUVs).astype(np.float32)/127.5 - 1.0
  # RGB: [0, 255], YUV: [0, 255]/127.5 - 1 = [-1, 1]

  print("#---  np.shape(CsYUVs) =", np.shape(CsYUVs))
  return CsYUVs

if __name__ == "__main__":
  all_dirs = os.listdir('/home/jinn/dataB')
  all_videos = ['/home/jinn/dataB/'+i+'/video.hevc' for i in all_dirs]
  for vi in all_videos:
    print("#---  video =", vi)
    fr = FrameReader(vi)
    frame_count = fr.frame_count
    print('#---  frame_count =', frame_count)
    cap = cv2.VideoCapture(vi)
    sYUVs = RGB_to_sYUVs(cap, frame_count)
    #---  np.shape(sYUVs) = (10, 384, 512)
    CsYUVs = sYUVs_to_CsYUVs(sYUVs)
    #---  np.shape(CsYUVs) = (10, 6, 128, 256)

    #for i in range(9):
    for i in range(len(CsYUVs) - 1):
      inputs = [np.vstack(CsYUVs[i:i+2])]
      #---  np.shape(inputs) = (1, 12, 128, 256)

      ret, frame = cap.read()
      #---  frame.shape = (874, 1164, 3) = (H, W, C)
      #frame = cv2.resize(frame, (640, 420))
      #---  frame.shape = (420, 640, 3) = (H, W, C)

      '''
      # Show RGB video images
      cv2.imshow("video (874x1164)", frame)
      cv2.waitKey(500)  # Wait a 0.5 second (for testing)

      # Convert YUV420 to Grayscale
      gray = cv2.cvtColor(sYUVs[i], cv2.COLOR_YUV2GRAY_I420)
      cv2.imshow("yuv2gray", gray)
      cv2.waitKey(500)  # Wait a 0.5 second (for testing)

      # Convert YUV420 to RGB (for testing), applies BT.601 "Limited Range" conversion.
      #rgb = cv2.cvtColor(CsYUVs[i], cv2.COLOR_YUV2RGB_I420)  # Error
      rgb = cv2.cvtColor(sYUVs[i], cv2.COLOR_YUV2RGB_I420)
      cv2.imshow("yuv2rgb (256x512)", rgb)
      cv2.waitKey(500)  # Wait a 0.5 second (for testing)

    input("Press ENTER twice to close all windows ...")
    input("Press ENTER to exit ...")
    # pauses for 1 second
    if cv2.waitKey(1000) == 27: #if ENTER is pressed
      cap.release()
      cv2.destroyAllWindows()
      '''

    print("#---  np.shape(inputs) =", np.shape(inputs))
