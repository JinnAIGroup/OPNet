'''  JLL, 2021.10.29
From /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py
Convert .png to .h5 for modelA4

Create folder /home/jinn/dataAll/comma10k/Ximgs (with two png files) and
 empty folder /home/jinn/dataAll/comma10k/Ximgs_yuv for debugging first
Create an empty folder /home/jinn/dataAll/comma10k/imgs_yuv
(OP082) jinn@Liu:~/openpilot/tools/lib$ python hevc2yuvh5A4.py
Input:
/home/jinn/dataAll/comma10k/imgs/*.png
Output:
/home/jinn/dataAll/comma10k/imgs_yuv/*_yuv.h5
'''
import os
import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from cameraB3 import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics

'''
big   RGB (874, 1164, 3) = (H, W, C) => big YUV (1311, 1164) = (X, Y) =>
small YUV (384, 512) => CsYUV (6, 128, 256) = (C, x, y) =>
small RGB (256, 512, 3)
'''
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def RGB_to_sYUV(img):
    bYUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
      #print("#---  np.shape(bYUV) =", np.shape(bYUV))
      #---  np.shape(bYUV) = (1311, 1164)

    sYUV = np.zeros((384, 512), dtype=np.uint8) # np.uint8 = 0~255
    sYUV = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                         yuv=True, output_size=(512, 256))  # (W, H)

    return sYUV

def sYUV_to_CsYUV(sYUV):
    H = (sYUV.shape[0]*2)//3  # 384x2//3 = 256
    W = sYUV.shape[1]         # 512
    CsYUV = np.zeros((6, H//2, W//2), dtype=np.uint8)

    CsYUV[0] = sYUV[0:H:2, 0::2]  # [2::2] get every even starting at 2
    CsYUV[1] = sYUV[1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
    CsYUV[2] = sYUV[0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
    CsYUV[3] = sYUV[1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
    CsYUV[4] = sYUV[H:H+H//4].reshape((-1, H//2, W//2))
    CsYUV[5] = sYUV[H+H//4:H+H//2].reshape((-1, H//2, W//2))

    CsYUV = np.array(CsYUV).astype(np.float32)/127.5 - 1.0
    # RGB: [0, 255], YUV: [0, 255]/127.5 - 1 = [-1, 1]??? Check /home/jinn/openpilot/selfdrive/camerad/transforms/rgb_to_yuv.cl
    # Check /selfdrive/modeld/models/dmonitoring.cc

    return CsYUV

def makeYUV(all_dirs):   # len(all_pngs) = 9888
    all_pngs = ['/home/jinn/dataAll/comma10k/Ximgs/'+i for i in all_dirs]   # debug
    #all_pngs = ['/home/jinn/dataAll/comma10k/imgs/'+i for i in all_dirs]
    print('#---  len(all_pngs) =', len(all_pngs))
    for png in all_pngs:
      yuvh5 = png.replace('.png','_yuv.h5')
      #yuvh5 = yuvh5.replace('Ximgs','Ximgs_yuv')
      yuvh5 = yuvh5.replace('imgs','imgs_yuv')
        #print("#---  png   =", png)
        #print("#---  yuvh5 =", yuvh5)
      if not os.path.isfile(yuvh5):
        img = cv2.imread(png)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #visualize(img)

        h5f = h5py.File(yuvh5, 'w')

        sYUV = RGB_to_sYUV(img)
          #---  np.shape(sYUV) = (384, 512)

        CsYUV = sYUV_to_CsYUV(sYUV)
          #---  np.shape(CsYUV) = (6, 128, 256)
        h5f.create_dataset('X', data=CsYUV)
        print("#---  np.shape(h5f) =", np.shape(h5f))
        print("#---  h5f =", h5f)
        h5f.close()

        print("#---  yuv.h5 created ...")
        yuvh5f = h5py.File(yuvh5, 'r') # read .h5, 'w': write
          #---  yuvh5f.keys() = <KeysViewHDF5 ['X']>
          #---  yuvh5f['X'].shape = (6, 128, 256)
          #---  yuvh5f['X'].dtype = float32
          #---  yuvh5f['X']       = <HDF5 dataset "X": shape (6, 128, 256), type "<f4">
      else:
        print("#---  yuv.h5 exists ...")
        yuvh5f = h5py.File(yuvh5, 'r') # read .h5, 'w': write
          #---  yuvh5f['X'].shape = (6, 128, 256)
          #print("#---  yuvh5f['X'][0]    =", yuvh5f['X'][0])

if __name__ == "__main__":
    all_dirs = os.listdir('/home/jinn/dataAll/comma10k/Ximgs')   # debug
    #all_dirs = os.listdir('/home/jinn/dataAll/comma10k/imgs')
    makeYUV(all_dirs)
