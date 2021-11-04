"""   YPL & JLL, 2021.11.2, 11.4
from /home/jinn/YPN/OPNet/datagenB3.py

Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=> CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=> CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging)
  /home/jinn/dataAll/comma10k/Xmasks/*.png
Output:
  X_batch.shape = (2, 12, 128, 256)
  Y_batch.shape = (2, 256, 512, 12)
"""
import os
import cv2
import h5py
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def visualize(image):
    #plt.figure(figsize=(10, 10))
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def concatenate(images, masks, mask_H, mask_W, class_values):
    all_images = []
    all_masks = []

    for img, msk in zip(images, masks):
        if os.path.isfile(img) and os.path.isfile(msk):
            imgH5 = h5py.File(img, 'r')
              #---  imgH5['X'].shape = (6, 128, 256)

            mskCV2 = cv2.imread(msk, 0).astype('uint8')  # https://github.com/YassineYousfi/comma10k-baseline/blob/main/retriever.py
            print('#---1  mskCV2.shape =', mskCV2.shape)
              #---1  mskCV2.shape = (874, 1164)
            visualize(mskCV2)

            mskCV2 = cv2.resize(mskCV2, (mask_W, mask_H))
            print('#---2  mskCV2.shape =', mskCV2.shape)
              #---2  mskCV2.shape = (256, 512)
            visualize(mskCV2)

            mskCV2 = np.stack([(mskCV2 == v) for v in class_values], axis=-1).astype('uint8')
            print('#---3  mskCV2.shape =', mskCV2.shape)
              #---3  mskCV2.shape = (256, 512, 6)

            all_images.append(imgH5['X'])
            all_masks.append(mskCV2)
        else:
            print('#---datagenA4  Error: image_yuv.h5 or mask.png does not exist')

    print('#---datagenA4  np.shape(all_images) =', np.shape(all_images))
    print('#---datagenA4  np.shape(all_masks) =', np.shape(all_masks))
      #---  np.shape(all_images) = (5, 6, 128, 256)
      #---  np.shape(all_masks)  = (5, 256, 512, 6)
    return all_images, all_masks

def datagen(images, masks, batch_size, image_H, image_W, mask_H, mask_W, class_values):
    all_images, all_masks = concatenate(images, masks, mask_H, mask_W, class_values)
    X_batch = np.zeros((batch_size, 12, image_H, image_W), dtype='uint8')
    Y_batch = np.zeros((batch_size, mask_H, mask_W, 2*len(class_values)), dtype='float32')
    imgsN = len(all_images)
    batchIndx = 0
    np.random.seed(0)

    while True:
        t = time.time()
        count = 0
        while count < batch_size:
            ri = np.random.randint(0, imgsN-1, 1)[-1]
            for i in range(imgsN-1):
                if ri < imgsN-1:   # the last imge is used only once
                    vsX1 = all_images[ri]
                    vsX2 = all_images[ri+1]
                    X_batch[count] = np.vstack((vsX1, vsX2))

                    vsY1 = all_masks[ri]
                    vsY2 = all_masks[ri+1]
                    Y_batch[count] = np.concatenate((vsY1, vsY1), axis=-1)
                break

            print('#---  count =', count)
            print('#---  not repeated? predictable? ri =', ri)
            count += 1

        batchIndx += 1
        print('#---  batchIndx =', batchIndx)
        t2 = time.time()
        print('#---  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))

        print('#---datagenA4  vsX2.shape =', vsX2.shape)
        print('#---datagenA4  vsY2.shape =', vsY2.shape)
        print('#---datagenA4  X_batch.shape =', X_batch.shape)
        print('#---datagenA4  Y_batch.shape =', Y_batch.shape)
          #---  X_batch.shape = (2, 12, 128, 256)
          #---  Y_batch.shape = (2, 256, 512, 6)
        yield(X_batch, Y_batch)
