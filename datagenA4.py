"""   YPL & JLL, 2021.11.2
from /home/jinn/YPN/OPNet/datagenB3.py
Input:
/home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging)
/home/jinn/dataAll/comma10k/Xmasks/*.png
Output:
X_batch.shape = (2, 12, 128, 256)
Y_batch.shape = (2, 256, 512, 6)
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
    Y_batch = np.zeros((batch_size, mask_H, mask_W, len(class_values)), dtype='float32')
    batchIndx = 0

    while True:
        t = time.time()
        count = 0

        while count < batch_size:
            for i in range(len(all_images)):
                #---  Xcf5.shape = (6, 128, 256)
              vsX1 = all_images[count]
              if (count+1) == batch_size:   # every img used twice except the first
                  vsX2 = all_images[count]
              else:
                  vsX2 = all_images[count+1]
                #print('#---  vsX2.shape =', vsX2.shape)
                #---  vsX2.shape = (6, 128, 256)
              X_batch[count] = np.vstack((vsX1, vsX2))

              #Y_batch[count] = np.vstack((vsPnL1, vsPnL2))
              break

            print('#---  count =', count)
            count += 1

        batchIndx += 1
        print('#---  batchIndx =', batchIndx)
        t2 = time.time()
        print('#---  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))

        print('#---datagenA4  X_batch.shape =', X_batch.shape)
        print('#---datagenA4  Y_batch.shape =', Y_batch.shape)
          #---  X_batch.shape = (2, 12, 128, 256)
          #---  Y_batch.shape = (2, 256, 512, 6)
        yield(X_batch, Y_batch)
