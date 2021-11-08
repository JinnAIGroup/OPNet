"""   YPL & JLL, 2021.11.8
from /home/jinn/YPN/OPNet/datagenA4a.py (see ???)

Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=> CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=> CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging)
  /home/jinn/dataAll/comma10k/Xmasks/*.png
Output:
  X_batch.shape = (none, 12, 128, 256)
  Y_batch.shape = (none, 256, 512, 12)
"""
import os
import cv2
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

def visualize(image):
    #plt.figure(figsize=(10, 10))
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def concatenate(img, msk, mask_H, mask_W, class_values):
    if os.path.isfile(img) and os.path.isfile(msk):
        imgH5X = h5py.File(img, 'r')
          #---  imgH5X['X'].shape = (6, 128, 256)
        imgH5 = imgH5X['X']
          #---  imgH5.shape = (6, 128, 256)

        mskCV2 = cv2.imread(msk, 0).astype('uint8')  # https://github.com/YassineYousfi/comma10k-baseline/blob/main/retriever.py
        #print('#---1  mskCV2.shape =', mskCV2.shape)
          #---1  mskCV2.shape = (874, 1164)
        #visualize(mskCV2)

        mskCV2 = cv2.resize(mskCV2, (mask_W, mask_H))
        #print('#---2  mskCV2.shape =', mskCV2.shape)
          #---2  mskCV2.shape = (256, 512)
        #visualize(mskCV2)

        mskCV2 = np.stack([(mskCV2 == v) for v in class_values], axis=-1).astype('uint8')
        #print('#---3  mskCV2.shape =', mskCV2.shape)
          #---3  mskCV2.shape = (256, 512, 6)
    else:
        print('#---datagenA4  Error: image_yuv.h5 or mask.png does not exist')

    return imgH5, mskCV2

def datagen(images, masks, batch_size, image_H, image_W, mask_H, mask_W, class_values):
    #X_batch = np.zeros((batch_size, 12, image_H, image_W), dtype='uint8')   # for RGB imgs
    X_batch = np.zeros((batch_size, 12, image_H, image_W), dtype='float32')   # for YUV imgs
    Y_batch = np.zeros((batch_size, mask_H, mask_W, 2*len(class_values)), dtype='float32')
    imgsN = len(images)
    print('#---datagenA4  imgsN =', imgsN)
    batchIndx = 0

    while True:
        t = time.time()
        count = 0
        #np.random.seed(0)
        while count < batch_size:
            ri = np.random.randint(0, imgsN-1, 1)[-1]
              #---  images[ri] = /home/jinn/dataAll/comma10k/Ximgs_yuv/0003_97a4ec76e41e8853_2018-09-29--22-46-37_5_585_yuv.h5
              #---  masks[ri] = /home/jinn/dataAll/comma10k/Xmasks/0003_97a4ec76e41e8853_2018-09-29--22-46-37_5_585.png
            for i in range(imgsN-1):
                if ri < imgsN-1:   # the last imge is used only once
                    vsX1, vsY1 = concatenate(images[ri], masks[ri], mask_H, mask_W, class_values)
                      #---  vsX1.shape, vsY1.shape = (6, 128, 256) (256, 512, 6)
                    vsX2, vsY2 = concatenate(images[ri+1], masks[ri+1], mask_H, mask_W, class_values)

                    X_batch[count] = np.vstack((vsX1, vsX2))
                    Y_batch[count] = np.concatenate((vsY1, vsY1), axis=-1)
                break

            #print('#---  count =', count)
            #print('#---  not repeated? predictable? ri =', ri)
            count += 1

        batchIndx += 1
        print('#---  batchIndx =', batchIndx)
        t2 = time.time()
        #print('#---  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))

        #print('#---datagenA4  X_batch.shape =', X_batch.shape)
        #print('#---datagenA4  Y_batch.shape =', Y_batch.shape)
          #---  X_batch.shape = (2, 12, 128, 256)
          #---  Y_batch.shape = (2, 256, 512, 12)
        #print('#---datagenA4  np.array(vsX1)[:,  64, 128] =', np.array(vsX1)[:, 64, 128])
        #print('#---datagenA4  X_batch[1,   :,  64, 128] =', X_batch[0, :, 64, 128])
        yield(X_batch, Y_batch)
