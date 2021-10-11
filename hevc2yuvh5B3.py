'''  JLL, 2021.9.24, 10.2, 10.9
From /home/jinn/openpilot/tools/lib/OPNet/video_to_yuvB3a.py
Convert video.hevc to yuv.h5

(OP082) jinn@Liu:~/openpilot/tools/lib$ python hevc2yuvh5B3.py
Input:
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/video.hevc
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/video.hevc
Output:
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/yuv.h5
yuv.h5        (256×512×3÷2 = 196608 x 1150 frames => 904.4 M)
camera320.h5  (160×320×3   = 155520 x 1150 frames => 706.6 M)
'''
import os
import sys
import cv2
import h5py
import numpy as np
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
  #for i in range(frame_count):
  for i in range(1150):   # 1150: see /home/jinn/YPN/YPNetB/datagenB3.py
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
  # RGB: [0, 255], YUV: [0, 255]/127.5 - 1 = [-1, 1]??? Check /home/jinn/openpilot/selfdrive/camerad/transforms/rgb_to_yuv.cl
  # Check /selfdrive/modeld/models/dmonitoring.cc

  print("#---  np.shape(CsYUVs) =", np.shape(CsYUVs))
  return CsYUVs

def makeYUV(all_dirs):
  all_videos = ['/home/jinn/dataB/'+i+'/video.hevc' for i in all_dirs]
  for vi in all_videos:
    yuvh5 = vi.replace('video.hevc','yuv.h5')
    print("#---  video =", vi)

    if not os.path.isfile(yuvh5):
    #if os.path.isfile(yuvh5):  # rewrite yuv.h5
      fr = FrameReader(vi)
      frame_count = fr.frame_count
      print('#---  frame_count =', frame_count)
      cap = cv2.VideoCapture(vi)

      with h5py.File(yuvh5, 'w') as h5f:
        #h5f.create_dataset('X', (frame_count, 6, 128, 256))
        h5f.create_dataset('X', (1150, 6, 128, 256))
        sYUVs = RGB_to_sYUVs(cap, frame_count)
        #---  np.shape(sYUVs) = (10, 384, 512)
        CsYUVs = sYUVs_to_CsYUVs(sYUVs)

        #for i in range(frame_count):
        for i in range(1150):   # 1150: see /home/jinn/YPN/YPNetB/datagenB3.py
          h5f['X'][i] = CsYUVs[i]

        print("#---  yuv.h5 created ...")
        yuvh5f = h5py.File(yuvh5, 'r') # read .h5, 'w': write
        print("#---  yuvh5f.keys() =", yuvh5f.keys())
        print("#---  yuvh5f['X'].shape =", yuvh5f['X'].shape)
        print("#---  yuvh5f['X'].dtype =", yuvh5f['X'].dtype)
        #print("#---  yuvh5f[0]    =", yuvh5f[0])
    else:
      print("#---  yuv.h5 exists ...")
      yuvh5f = h5py.File(yuvh5, 'r') # read .h5, 'w': write
      print("#---  yuvh5f.keys() =", yuvh5f.keys())
      print("#---  yuvh5f['X'].shape =", yuvh5f['X'].shape)
      print("#---  yuvh5f['X'].dtype =", yuvh5f['X'].dtype)
      #print("#---  yuvh5f[0]    =", yuvh5f[0])
      #---  yuvh5f['X'].shape = (1199, 6, 128, 256)

if __name__ == "__main__":
  all_dirs = os.listdir('/home/jinn/dataB')
  makeYUV(all_dirs)
